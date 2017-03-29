import luigi
import os
from subprocess import Popen, PIPE


class BaseTask(luigi.Task):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    output_root = luigi.Parameter(default=output_dir)

    def ensure_output_dir(self):
        if not os.path.exists(os.path.dirname(self.output().path)):
            os.makedirs(os.path.dirname(self.output().path))
        return True


class ConsoleTask(BaseTask):

    def console(self, cmd):
        p = Popen(cmd, shell=True, stdout=PIPE)
        out, err = p.communicate()
        return p.returncode, out, err
    pass


class InputData(luigi.ExternalTask):
    file = luigi.Parameter()

    def output(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.file)
        return luigi.LocalTarget(path)

