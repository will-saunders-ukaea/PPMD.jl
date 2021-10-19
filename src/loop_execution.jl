export execute


"""
Generic container for a callable object that can be passed to execute.
"""
mutable struct Task
    name::String
    execute
    call_count::Int64
    runtime::Float64
    data::Dict
    function Task(name, func)
        return new(name, func, 0, 0.0, Dict())
    end
end


"""
Runs a single task and accumulates the total runtime and call count if the task
is of type Task.
"""
function run_task(task)

    if typeof(task) <: Task
        t = @elapsed begin
            task.execute()
        end
        task.runtime += t
        task.call_count += 1
    else
        task.execute()
    end

end


"""
Synchronously execute one or more tasks in order.
"""
function execute(task)
    
    if hasproperty(task, :execute)
        run_task(task)
    else
        for tx in task
            run_task(tx)
        end
    end

end
