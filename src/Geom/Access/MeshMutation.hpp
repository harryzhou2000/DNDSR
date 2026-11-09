#pragma once

#include "DNDS/Runtime/Storage/Version.hpp"
#include <exception>
#include <functional>
#include <thread>
#include <utility>

namespace DNDS::Geom
{
    /// Explicit capability for serialized mesh implementation calls. Public
    /// entrypoints always acquire a new writer; only private Impl methods join
    /// this token. A nested failure remains sticky even if its caller catches it.
    class MeshMutationToken
    {
        const void *_owner;
        std::thread::id _thread = std::this_thread::get_id();
        Runtime::VersionTracker::Mutation _mutation;
        size_t _depth = 0;
        bool _failed = false, _closed = false;

        // Never expose a movable parent borrow. Only Run constructs this guard,
        // and its destruction is part of that synchronous call.
        class Step
        {
            MeshMutationToken *_token;
            int _uncaught = std::uncaught_exceptions();
            friend class MeshMutationToken;
            explicit Step(MeshMutationToken &token) : _token(&token) { ++token._depth; }

        public:
            Step(const Step &) = delete;
            Step &operator=(const Step &) = delete;
            Step(Step &&) = delete;
            ~Step()
            {
                if (_token)
                {
                    _token->_failed |= std::uncaught_exceptions() > _uncaught;
                    --_token->_depth;
                }
            }
        };

    public:
        MeshMutationToken(const void *owner, Runtime::VersionTracker &versions, Runtime::Change changes,
                          const std::function<void()> &onBegin = {})
            : _owner(owner), _mutation(versions.BeginMutation(changes))
        {
            // Native query authority may be suspended only after this writer
            // has excluded active managed readers. Nested steps never repeat it.
            if (onBegin)
                onBegin();
        }
        MeshMutationToken(const MeshMutationToken &) = delete;
        MeshMutationToken &operator=(const MeshMutationToken &) = delete;
        ~MeshMutationToken()
        {
            // Destroying the token from inside its own callback is a contract
            // violation; fail before a guard can dereference a dead parent.
            if (_depth || std::this_thread::get_id() != _thread)
                std::terminate();
        }
        /// Execute a nested implementation synchronously. The callback must not
        /// destroy this token or schedule work that borrows it. Nested failure
        /// is sticky even when an outer callback catches the exception.
        template <class F>
        void Run(const void *owner, Runtime::Change changes, F &&implementation)
        {
            DNDS_check_throw_info(!_closed && !_failed && owner == _owner && std::this_thread::get_id() == _thread,
                                  "invalid, failed or foreign mesh mutation capability");
            _mutation.IncludeChanges(changes);
            Step step(*this);
            std::forward<F>(implementation)();
        }
        void Commit()
        {
            DNDS_check_throw_info(!_closed && !_failed && !_depth && std::this_thread::get_id() == _thread,
                                  "mesh mutation cannot commit an active or failed nested edit");
            _mutation.Commit();
            _closed = true;
        }
    };
}
