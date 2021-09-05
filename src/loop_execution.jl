export execute

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


function execute(task)
    
    if hasproperty(task, :execute)
        run_task(task)
    else
        for tx in task
            run_task(tx)
        end
    end

end
