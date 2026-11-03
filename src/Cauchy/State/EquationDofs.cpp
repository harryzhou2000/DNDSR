#include "Cauchy/State/EquationDofs.hpp"
#include <limits>
#include <algorithm>
#include <utility>

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    struct EquationDofs::Definition
    {
        struct Block
        {
            index field, first, width;
        };
        EquationSchema schema;
        Algebra::DofSpace space;
        R::Device device;
        std::vector<std::string> fields;
        std::vector<index> widths;
        std::vector<Block> blocks;
        index components = 0;
        bool coupled = false;

        index Find(const std::string &name) const
        {
            for (index iBlock = 0; iBlock < index(schema.blocks.size()); ++iBlock)
                if (schema.blocks[iBlock].name == name)
                    return iBlock;
            DNDS_check_throw_info(false, "unknown equation DOF block: " + name);
            return -1;
        }
        Definition(EquationSchema value, const Algebra::DofBundle &storage, const std::vector<EquationDofBinding> &bindings)
            : schema(std::move(value)), space(storage.Space()), device(R::Device::Host())
        {
            storage.Validate();
            DNDS_check_throw_info(!schema.family.empty() && !schema.physicalAccumulation.empty() && !schema.blocks.empty() &&
                                      storage.BlockNumber() > 0 && bindings.size() == schema.blocks.size(),
                                  "incomplete equation DOF schema/bindings");
            for (index iField = 0; iField < storage.BlockNumber(); ++iField)
            {
                fields.push_back(storage.Name(iField));
                const auto &field = storage.Block(fields.back());
                DNDS_check_throw_info(field.Width() <= size_t(std::numeric_limits<index>::max()), "equation DOF width overflows index");
                widths.push_back(index(field.Width()));
                if (!iField)
                    device = field.Device();
                DNDS_check_throw_info(field.Device() == device, "equation DOF fields use different devices");
            }
            blocks.resize(schema.blocks.size());
            std::vector<unsigned char> assigned(schema.blocks.size(), 0);
            std::vector<std::vector<std::pair<index, index>>> intervals(fields.size());
            for (index iBlock = 0; iBlock < index(schema.blocks.size()); ++iBlock)
            {
                const auto &block = schema.blocks[iBlock];
                DNDS_check_throw_info(!block.name.empty() && !block.components.empty() && block.components.size() <= size_t(std::numeric_limits<index>::max()), "empty/oversized equation block");
                for (index before = 0; before < iBlock; ++before)
                    DNDS_check_throw_info(schema.blocks[before].name != block.name, "duplicate equation block name");
                for (index iComponent = 0; iComponent < index(block.components.size()); ++iComponent)
                {
                    DNDS_check_throw_info(!block.components[iComponent].name.empty(), "empty equation component name");
                    for (index before = 0; before < iComponent; ++before)
                        DNDS_check_throw_info(block.components[before].name != block.components[iComponent].name, "duplicate equation component name");
                }
                for (index iVector = 0; iVector < index(block.vectorStarts.size()); ++iVector)
                {
                    const auto start = block.vectorStarts[iVector];
                    DNDS_check_throw_info(start >= 0 && index(block.components.size()) >= 3 && start <= index(block.components.size()) - 3, "equation vector range exceeds its block");
                    for (index before = 0; before < iVector; ++before)
                        DNDS_check_throw_info(std::abs(start - block.vectorStarts[before]) >= 3, "overlapping equation vector ranges");
                }
                DNDS_check_throw_info(index(block.components.size()) <= std::numeric_limits<index>::max() - components, "equation component count overflows");
                components += index(block.components.size());
            }
            for (const auto &binding : bindings)
            {
                const index iBlock = Find(binding.block);
                DNDS_check_throw_info(!assigned[iBlock], "duplicate equation DOF binding");
                assigned[iBlock] = 1;
                const auto field = std::find(fields.begin(), fields.end(), binding.field);
                DNDS_check_throw_info(field != fields.end(), "unknown physical equation DOF field: " + binding.field);
                const index iField = field - fields.begin(), width = index(schema.blocks[iBlock].components.size());
                DNDS_check_throw_info(binding.first >= 0 && width <= widths[iField] && binding.first <= widths[iField] - width, "equation DOF binding exceeds physical field");
                blocks[iBlock] = {iField, binding.first, width};
                intervals[iField].emplace_back(binding.first, binding.first + width);
            }
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                auto &ranges = intervals[iField];
                std::sort(ranges.begin(), ranges.end());
                index end = 0;
                for (const auto &range : ranges)
                {
                    DNDS_check_throw_info(range.first == end, "equation DOF coverage overlaps or leaves a gap");
                    end = range.second;
                }
                DNDS_check_throw_info(end == widths[iField], "equation DOF coverage omits physical components");
            }
            coupled = fields.size() == 1;
            index first = 0;
            for (const auto &block : blocks)
            {
                coupled = coupled && block.field == 0 && block.first == first;
                first += block.width;
            }
        }
    };
    void EquationDofs::PinFields()
    {
        _pins.clear();
        _pins.reserve(size_t(_fields.BlockNumber()));
        for (index iField = 0; iField < _fields.BlockNumber(); ++iField)
        {
            const auto &field = _fields.Block(_fields.Name(iField));
            _pins.push_back({ArrayPin{field.Owned().SharedStorage(), field.Owned().Identity(), field.Owned().ShapeVersion()},
                             ArrayPin{field.Ghost().SharedStorage(), field.Ghost().Identity(), field.Ghost().ShapeVersion()}});
        }
    }
    EquationDofs::EquationDofs(EquationSchema schema, Algebra::DofBundle fields, std::vector<EquationDofBinding> bindings)
        : _definition(std::make_shared<const Definition>(std::move(schema), fields, bindings)), _fields(std::move(fields))
    {
        PinFields();
        Validate();
    }
    EquationDofs::EquationDofs(std::shared_ptr<const Definition> definition, Algebra::DofBundle fields)
        : _definition(std::move(definition)), _fields(std::move(fields))
    {
        PinFields();
        Validate();
    }
    EquationDofs EquationDofs::Coupled(EquationSchema schema, Algebra::DofField<real> field, std::string name)
    {
        std::vector<EquationDofBinding> bindings;
        index first = 0;
        for (const auto &block : schema.blocks)
        {
            DNDS_check_throw_info(block.components.size() <= size_t(std::numeric_limits<index>::max() - first), "equation component count overflows");
            bindings.push_back({block.name, name, first});
            first += index(block.components.size());
        }
        Algebra::DofBundle fields(field.Space());
        fields.Add(std::move(name), std::move(field));
        return {std::move(schema), std::move(fields), std::move(bindings)};
    }
    EquationDofs EquationDofs::Separate(EquationSchema schema, Algebra::DofBundle fields)
    {
        std::vector<EquationDofBinding> bindings;
        for (const auto &block : schema.blocks)
            bindings.push_back({block.name, block.name, 0});
        return {std::move(schema), std::move(fields), std::move(bindings)};
    }
    const EquationSchema &EquationDofs::Schema() const
    {
        DNDS_check_throw_info(_definition, "equation DOF binding is moved");
        return _definition->schema;
    }
    index EquationDofs::ComponentNumber() const
    {
        DNDS_check_throw_info(_definition, "equation DOF binding is moved");
        return _definition->components;
    }
    index EquationDofs::BlockNumber() const { return index(Schema().blocks.size()); }
    R::Device EquationDofs::Device() const
    {
        DNDS_check_throw_info(_definition, "equation DOF binding is moved");
        return _definition->device;
    }
    const Algebra::DofField<real> &EquationDofs::PhysicalField(index iField) const
    {
        Validate();
        DNDS_check_throw_info(iField >= 0 && iField < _fields.BlockNumber(), "physical equation field index out of range");
        return _fields.Block(_fields.Name(iField));
    }
    Algebra::DofField<real> &EquationDofs::PhysicalField(index iField)
    {
        return const_cast<Algebra::DofField<real> &>(std::as_const(*this).PhysicalField(iField));
    }
    EquationBlockLocation EquationDofs::LocateBlock(index iBlock) const
    {
        Validate();
        DNDS_check_throw_info(iBlock >= 0 && iBlock < index(_definition->blocks.size()), "equation block index out of range");
        const auto &block = _definition->blocks[iBlock];
        return {block.field, block.first, block.width};
    }
    EquationBlockLocation EquationDofs::LocateBlock(const std::string &block) const
    {
        DNDS_check_throw_info(_definition, "equation DOF binding is moved");
        return LocateBlock(_definition->Find(block));
    }
    void EquationDofs::Validate() const
    {
        DNDS_check_throw_info(_definition && _fields.Space().Same(_definition->space) && _fields.BlockNumber() == index(_definition->fields.size()) &&
                                  _pins.size() == _definition->fields.size(),
                              "equation DOF binding is moved or its row space changed");
        _fields.Validate();
        for (index iField = 0; iField < _fields.BlockNumber(); ++iField)
        {
            DNDS_check_throw_info(_fields.Name(iField) == _definition->fields[iField], "equation DOF physical field ordering changed");
            const auto &field = _fields.Block(_fields.Name(iField));
            DNDS_check_throw_info(index(field.Width()) == _definition->widths[iField] && field.Device() == _definition->device, "equation DOF physical shape/device changed");
            const std::array<const R::Array<real> *, 2> arrays{&field.Owned(), &field.Ghost()};
            for (int part = 0; part < 2; ++part)
            {
                const auto &pin = _pins[iField][part];
                DNDS_check_throw_info(arrays[part]->SharedStorage() == pin.storage && arrays[part]->Identity() == pin.identity && arrays[part]->ShapeVersion() == pin.shape,
                                      "equation DOF storage/shape changed; bind the replacement explicitly");
            }
        }
    }
    void EquationDofs::RequireSchema(const EquationSchema &schema) const
    {
        Validate();
        const auto &actual = Schema();
        bool same = actual.family == schema.family && actual.energyConvention == schema.energyConvention && actual.physicalAccumulation == schema.physicalAccumulation && actual.blocks.size() == schema.blocks.size();
        for (index iBlock = 0; same && iBlock < index(actual.blocks.size()); ++iBlock)
        {
            const auto &a = actual.blocks[iBlock], &b = schema.blocks[iBlock];
            same = a.name == b.name && a.vectorStarts == b.vectorStarts && a.components.size() == b.components.size();
            for (index iComponent = 0; same && iComponent < index(a.components.size()); ++iComponent)
                same = a.components[iComponent].name == b.components[iComponent].name && a.components[iComponent].units == b.components[iComponent].units && a.components[iComponent].conservationGroup == b.components[iComponent].conservationGroup;
        }
        DNDS_check_throw_info(same, "equation DOF semantic schema differs from the operator");
    }
    void EquationDofs::RequireCompatible(const EquationDofs &other) const
    {
        RequireSchema(other.Schema());
        other.Validate();
        DNDS_check_throw_info(Space().Same(other.Space()) && Device() == other.Device(), "equation DOF row space/device differs");
    }
    const Algebra::DofField<real> &EquationDofs::CoupledField() const
    {
        Validate();
        DNDS_check_throw_info(_definition->coupled, "this stage adapter requires one field in coupled logical order; split/permuted storage is unsupported");
        return _fields.Block(_definition->fields.front());
    }
    Algebra::DofField<real> &EquationDofs::CoupledField()
    {
        return const_cast<Algebra::DofField<real> &>(std::as_const(*this).CoupledField());
    }
    EquationDofs EquationDofs::Clone() const
    {
        Validate();
        return {_definition, _fields.Clone()};
    }
    EquationDofs EquationDofs::RebindCoupled(Algebra::DofField<real> field) const
    {
        Validate();
        DNDS_check_throw_info(_definition->coupled, "stage rebinding requires coupled logical storage");
        Algebra::DofBundle fields(field.Space());
        fields.Add(_definition->fields.front(), std::move(field));
        return {_definition, std::move(fields)};
    }
    EquationDofs EquationDofs::RebindFields(Algebra::DofBundle fields) const
    {
        Validate();
        return {_definition, std::move(fields)};
    }
    EquationDofs::ReadScope::ReadScope(const EquationDofs &state, bool ghosts, R::ExecutionContext context)
        : _definition(state._definition)
    {
        state.Validate();
        DNDS_check_throw_info(context.device == state.Device(), "equation DOF read context uses the wrong device");
        _owned.reserve(size_t(state._fields.BlockNumber()));
        if (ghosts)
            _ghost.reserve(size_t(state._fields.BlockNumber()));
        try
        {
            for (index iField = 0; iField < state._fields.BlockNumber(); ++iField)
            {
                const auto &field = state._fields.Block(state._fields.Name(iField));
                _owned.emplace_back(field.Owned().Read(context));
                if (ghosts)
                    _ghost.emplace_back(field.Ghost().Read(context));
            }
        }
        catch (...)
        {
            for (auto &lease : _owned)
                lease.CancelUnmodified();
            for (auto &lease : _ghost)
                lease.CancelUnmodified();
            throw;
        }
    }
    EquationBlockReadView EquationDofs::ReadScope::Block(index iBlock) const
    {
        DNDS_check_throw_info(_definition && iBlock >= 0 && iBlock < index(_definition->blocks.size()), "equation block index out of range or moved scope");
        const auto &block = _definition->blocks[iBlock];
        EquationBlockReadView view;
        view.owned = {_owned[block.field].View<R::UniformTableRows>(), block.first, block.width};
        view.ghost = {{}, block.first, block.width};
        if (!_ghost.empty())
            view.ghost.parent = _ghost[block.field].View<R::UniformTableRows>();
        return view;
    }
    EquationBlockReadView EquationDofs::ReadScope::Block(const std::string &name) const
    {
        DNDS_check_throw_info(_definition, "equation DOF scope is moved");
        return Block(_definition->Find(name));
    }
    void EquationDofs::ReadScope::Complete(const R::Completion &completion)
    {
        for (auto &lease : _owned)
            lease.Complete(completion);
        for (auto &lease : _ghost)
            lease.Complete(completion);
    }
    EquationDofs::WriteScope::WriteScope(EquationDofs &state, R::ExecutionContext context)
        : _definition(state._definition)
    {
        state.Validate();
        DNDS_check_throw_info(context.device == state.Device(), "equation DOF write context uses the wrong device");
        _owned.reserve(size_t(state._fields.BlockNumber()));
        try
        {
            for (index iField = 0; iField < state._fields.BlockNumber(); ++iField)
                _owned.emplace_back(state._fields.Block(state._fields.Name(iField)).Owned().Write(context));
        }
        catch (...)
        {
            CancelUnmodified();
            throw;
        }
    }
    EquationBlockArrayView<real> EquationDofs::WriteScope::Block(index iBlock) const
    {
        DNDS_check_throw_info(_definition && iBlock >= 0 && iBlock < index(_definition->blocks.size()), "equation block index out of range or moved scope");
        const auto &block = _definition->blocks[iBlock];
        return {_owned[block.field].View<R::UniformTableRows>(), block.first, block.width};
    }
    EquationBlockArrayView<real> EquationDofs::WriteScope::Block(const std::string &name) const
    {
        DNDS_check_throw_info(_definition, "equation DOF scope is moved");
        return Block(_definition->Find(name));
    }
    void EquationDofs::WriteScope::Complete(const R::Completion &completion)
    {
        for (auto &lease : _owned)
            lease.Complete(completion);
    }
    void EquationDofs::WriteScope::CancelUnmodified()
    {
        for (auto &lease : _owned)
            lease.CancelUnmodified();
    }
    EquationDofs::ReadScope EquationDofs::ReadOwned(R::ExecutionContext context) const { return {*this, false, context}; }
    EquationDofs::ReadScope EquationDofs::ReadWithGhost(R::ExecutionContext context) const { return {*this, true, context}; }
    EquationDofs::WriteScope EquationDofs::WriteOwned(R::ExecutionContext context) { return {*this, context}; }
}
