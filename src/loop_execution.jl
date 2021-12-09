export execute


"""
Generic container for a callable object that can be passed to execute.
"""
mutable struct Task
    name::String
    execute
    call_count::Int64
    runtime::Float64
    runtime_inner::Float64
    data::Dict
    function Task(name, func)
        return new(name, func, 0, 0.0, 0.0, Dict())
    end
end


"""
Runs a single task and accumulates the total runtime and call count if the task
is of type Task.
"""
function run_task(task)

    t = @elapsed begin
        task.execute()
    end
    
    name = "anon_run_task"
    if typeof(task) <: Task
        task.runtime += t
        task.call_count += 1
        name = task.name
    end

    increment_profiling_value(name, "time" , t)
    increment_profiling_value(name, "count" , 1)

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
