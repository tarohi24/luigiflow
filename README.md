luigiflow
=====

luigiflow is a simple machine learning task manager.
luigiflow is built on two popular Python frameworks: luigi and mlflow.
For more detail explanation about this framework, take a look at my blog post:
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
set -o errexit
set -o nounset
set -o pipefail

DB_URI="sqlite:///db.sqlite3"  # Specify backend database
ARTIFACTS_DIR="/PATH/TO/ARTIFACTS_DIR"  # Specify a directory where mlflow saves arfifacts  
PORT=8000

mlflow server \
    --backend-store-uri ${DB_URI} \
    --port ${PORT} \
    --host 0.0.0.0 \
    --default-artifact-root ${ARTIFACTS_DIR}
```

3. Launch a luigi server. If you run it on your local machine, run `$ luigid`.
4. Implement a task. For example, the following task is a task just to print "hello".
   For more detailed explanations about how to implement a task, see the following section.

```python
from luigiflow.task import MlflowTask

class HelloTask(MlflowTask):

    @classmethod
    def get_experiment_name(cls):
       return "hello"
    
    @classmethod
    def get_artifact_filenames(cls):
       return dict()
    
    def requires(self):
       return dict()
    
    def _run(self):
       print('hello')

```
5. Run a task using `luigi.bulid()`. For example, you can run the `HelloTask` as follows.

```python
import luigi
import mlflow

mlflow.set_tracking_uri("http://localhost:8000")  # Update the URI according to your mlflow config

luigi.build([HelloTask(), ])
```
