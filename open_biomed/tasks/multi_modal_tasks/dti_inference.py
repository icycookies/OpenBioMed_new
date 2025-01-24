from open_biomed.core.pipeline import InferencePipeline
from open_biomed.core.pipeline import TrainValPipeline
from open_biomed.data import Molecule, Text
from open_biomed.data.molecule import MoleculeDataModule
from open_biomed.tasks.multi_modal_tasks import DTIInferencePipeline, DTITrainingPipeline

if __name__ == "__main__":
    pipeline=TrainValPipeline()
    pipeline.run()

