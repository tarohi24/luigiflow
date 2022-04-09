luigiflow
=====

luigiflow is a simple machine learning task manager.
luigiflow is built on two popular Python frameworks: luigi and mlflow.
For more explanations about this framework, take a look at my blog post:
[Toward the minimalism of machine learning experiment workflows](https://blog.whiro.me/minimal-ml-experiment-workflows/)

I've implemented luigiflow to
- enhance integration between luigi and mlflow
- support strong type-hinting
- support jsonnet to specify both parameters of tasks and their dependencies.


## Installation

If you use pip, run the following command.

```bash
$ pip install git+https://github.com/tarohi24/luigiflow.git
```

## Get started!

### Outline
1. Launch an mlflow server.
2. Implement an `MlflowTaskProtocol` to denote the outputs (interface) of a task.
3. Implement an `MlflowTask`.


### Take a closer look at each step

**1. Launch an mlflow server.**
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

**2. Implement an `MlflowTaskProtocol`.**

Suppose that you implement a task that does pre-process of texts.
Whenever you implement a task, think of its outputs at first. 
The output of that task is an `Iteralbe[str]`, so you can implement `OutputTextListProtocol` as follows.
If you need a task to load texts to process, you'll also need `LoadTexts` protocol.


```python
from typing import Iterable, Protocol, runtime_checkable

from luigiflow.task import MlflowTaskProtocol


# you have to have `Protocol` as a base class to define it as a protocol
class OutputTextList(MlflowTaskProtocol, Protocol):

    # you can name arbitrarily.
    def get_preprocessed_texts(self) -> Iterable[str]:
        ...


class LoadTexts(MLflowTaskProtocol, Protocol):
    def load_texts(self) -> Iteralbe[str]:
        ...
```


**3. Implement a task**

Now you can implement a task.
Each task inherits from `MlflowTask`.
Let's begin with a task that loads texts from a CSV file.


```python
import pandas as pd

from .protocols import LoadTexts


class LoadTextFromFile(MlflowTask):
    # 1. denote parameters
    path: str = luigi.Parameter()
    text_column: str = luigi.Parameter(default="text")
    # 2. specify task config
    config = TaskConfig(
        experiment_name="load_texts",  # give an mlflow's experiment name
        protocols=[LoadTexts, ],  # protocols that this task implements
        # declares files that this task outputs (you don't need to specify full-paths, but just filenames)
        # because luigiflow automatically creates an artifact directory
        artifact_filenames={
            "texts": "texts.txt",
        },
    )

    # 3. Implement the main procedures of this task
    def _run(self):

        def save_texts(texts: list[str], path: str):
            with open(path, "w") as fout:
                fout.write("\n".join(texts))

        # load texts
        df = pd.read_csv(self.path)
        texts = df[self.text_column].tolist()
        # save outputs
        self.save_to_mlflow(
            # specify (object, save_fn) for each artifact
            artifacts_and_save_funcs={
                "texts": (texts, self.save_texts),
            },
        )

    # 4. Specify how to load artifacts (follow the types of protocols of this task)
    def load_texts(self) -> list[str]:
        assert self.complete()  # to load texts, this task needs to be completed
        output_path = self.output()["texts"]  # the key `texts` is specified at `config.artifact_filenames`
        with open(output_path) as fin:
            texts = fin.read().splitlines()
        return texts
```

*TODO: further explanations*

Then move to `PreprocessTexts`.


```python
from typing import Iterable, TypedDict

import luigi
from luigiflow.task import MlflowTask, TaskConfig

from .protocols import LoadTexts, OutputTextList


# if a task has required tasks, make a `TypedDict` to denote their protocols.
# You cannot specify `MlflowTask` as a requirement. You have to specify their protocols, instead.
class Requirements(TypedDict):
    load_texts: LoadTexts


# Pass the requirements type to the type argument of `MlflowTask` to activate type-hinting.
class PreProcessTexts(MlflowTask[Requirements]):
    config = TaskConfig(
        experiment_name="load_texts",
        protocols=[OutputTextList, LoadTexts],  # note that this task is an instance of `LoadTexts` as well.
        artifact_filenames={
            "texts": "texts.txt",
        },
        # specify requirements.
        # based on this value, luigiflow yields `task.requiers()` method.
        requirements={
            "load_texts": LoadText,
        }
    )

    def load_texts(self) -> Iterable[str]:
        # for LoadTexts
        with open(self.output()["texts"].path) as fin:
            texts = fin.read().splitlines()
        return texts

    def get_preprocessed_texts(self) -> Iterable[str]:
        # for OutputTextList
        return self.load_texts()

    def _run(self):
        def save_texts(texts: Iterable[str], path: str):
            with open(path, "w") as fout:
                fout.write("\n".join(texts))

        original_texts = self.requires()["load_texts"].load_texts()
        processed_texts = [
            text.lower() for text in original_texts
        ]
        self.save_to_mlflow(
            artifacts_and_save_funcs={
                "texts": (processed_texts, save_texts),
            }
        )

```

**4. Specify parameter values and run tasks**

Well done! Now that you've implemented all the tasks.
You can make a jsonnet file to specify parameter values, and then run tasks.

```jsonnet
{
    cls: "PreProcessTexts",
    params: {},
    requires: {
        load_texts: {
            cls: "LoadTextFromFile",
            params: {
                path: "data.csv",
            },
            requires: {},
        }
    }
}
```

(WIP)
```bash
$ luigiflow run OutputTextList --config config.jsonnet
```