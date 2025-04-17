SELECT * FROM run
JOIN task ON task.id = run.task_id
JOIN graph ON graph.id = run.graph_id
ORDER BY graph.id