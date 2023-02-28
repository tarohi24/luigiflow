```mermaid
flowchart
    T(tag_params.py) --> Ser(Serializer.py)
    Col(collection.py) --> Task(task.py)
    Col --> T
    Col --> Ser

    Run(task_run.py) --> Task
    Run --> T

```
