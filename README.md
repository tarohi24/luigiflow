luigiflow
=====

luigiflow is a simple machine learning task manager. luigiflow is built on two popular Python frameworks: luigi and
mlflow. For more detail explanation about this framework, take a look at my blog post:
[Toward the minimalism of machine learning experiment workflows](https://blog.whiro.me/minimal-ml-experiment-workflows/)

## Get started!

### Install this package

If you use pip, run the following command.

```bash
$ pip install git+https://github.com/tarohi24/luigiflow.git
```

2. Launch an mlflow server. If you start a server on your local machine, use the following command.

```bash
#!/bin/bash 
DB_URI="sqlite:///db.sqlite3"  # Specify backend database
ARTIFACTS_DIR="/PATH/TO/ARTIFACTS_DIR"  # Specify a directory where mlflow saves arfifacts  
PORT=8000

mlflow server \
    --backend-store-uri ${DB_URI} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --default-artifact-root ${ARTIFACTS_DIR}
```

3. Implement a protocol class. A protocol class abstracts tasks that have the same output. Each task is an
   implementation of more than one protocol. For example, you can define `OutputTextList` protocol.

```python
from typing import Iterable, Protocol, runtime_checkable

from luigiflow.task import MlflowTaskProtocol


# you have to have `Protocol` as a base to define it as a protocol
@runtime_checkable
class OutputTextList(MlflowTaskProtocol, Protocol):

    def load_texts(self) -> Iterable[str]:
        raise NotImplementedError
```

* Note that you cannot write any implementations on a protocol. A protocol is just to declare desirable behaviors.
* Don't forget to add `@runtime_checkable` when declaring a protocol. That's necessary when luigiflow checks if a task
  meets requirements of its protocols.

5. Let's implement a class.

```python
from typing import Iterable

import luigi
from luigiflow.task import MlflowTask, TaskConfig


class SayHelloClass(MlflowTask):
    message: str = luigi.Parameter()
    config = TaskConfig(
        experiment_name="hello",
        protocols=[OutputTextList, ],
        artifact_filenames={
            "texts": "texts.txt",
        },
    )

    def load_texts(self) -> Iterable[str]:
        with open(self.output()["texts"].path) as fin:
            texts = fin.read().splitlines()
        return texts

    def requires(self):
        return dict()
    
    @staticmethod
    def save_texts(texts: Iterable[str], path: str):
        with open(path, "w") as fout:
            fout.write("\n".join(texts))
            fout.write("\n")

    def _run(self):
        texts = [self.message, ]
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "texts": (texts, self.save_texts),
            }
        )
```

4. (necessary only if your task has parameters) Prepare a jsonnet file to set parameter values. Let's create "
   config.jsonnet", as follows.

```jsonnet
# config.jsonnet
local message = "good morning!";
{
  "HelloTask": {
    "message": message,
  }
}
```

(WIP)