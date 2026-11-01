#pragma once

#include "DNDS/Runtime/Storage/Buffer.hpp"

namespace DNDS::Runtime::Detail
{
    /// Operation-side immediate or caller-ordered scheduled access. Shared
    /// by MPI and stream-native providers; it does not infer stream edges.
    struct TransferAccess
    {
        ReadAccess read;
        WriteAccess write;
        ScheduledAccess pending;
        Completion completion;
        bool scheduled = false, writable = false;
        Device device;
        ExecutionContext context;
        void Acquire(Buffer &buffer, ExecutionContext execution, bool wr, bool sch)
        {
            context = execution;
            scheduled = sch;
            writable = wr;
            device = buffer.GetDevice();
            if (sch)
                pending = buffer.Reserve(wr ? AccessMode::Write : AccessMode::Read, context.device);
            else if (wr)
                write = buffer.Write(context);
            else
                read = buffer.Read(context);
        }
        template <class T>
        Span<T> View()
        {
            if (scheduled)
                return {static_cast<T *>(pending.Pointer(alignof(T), sizeof(T))), 0, device.space};
            if (writable)
                return write.View<T>();
            auto view = read.View<T>();
            return {const_cast<T *>(view.data), view.size, view.space};
        }
        void Complete(Completion event)
        {
            completion = std::move(event);
            if (!scheduled && device.space == MemorySpace::CUDA)
            {
                if (writable)
                    write.Complete(completion);
                else
                    read.Complete(completion);
            }
        }
        void Wait()
        {
            completion.Wait();
            if (scheduled)
                pending.Complete(true);
            else if (writable)
                write.Wait();
            else
                read.Wait();
        }
        void Fail()
        {
            std::exception_ptr error;
            try
            {
                if (scheduled && device.space == MemorySpace::CUDA)
                    Completion::Record(context).Wait();
                else
                    completion.Wait();
            }
            catch (...)
            {
                error = std::current_exception();
            }
            if (scheduled)
                pending.Complete(false);
            else if (writable)
                write.Fail();
            else
                read.Wait();
            if (error)
                std::rethrow_exception(error);
        }
        void CancelUnmodified()
        {
            if (scheduled)
                pending.Cancel();
            else if (writable)
                write.CancelUnmodified();
            else
                read.CancelUnmodified();
        }
    };
}
