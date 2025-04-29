from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

base_model = GroundedSAM(
    ontology=CaptionOntology(
        {
            "car": "car"
        }
    )
)

# run inference on a single image
results = base_model.predict("/Users/krishnaiyer/generative-ai-agentic-cv-base/data/processed/DAWN/Fog/foggy-003.jpg")

plot(
    image=cv2.imread("/Users/krishnaiyer/generative-ai-agentic-cv-base/data/raw/DAWN/Fog/foggy-003.jpg"),
    classes=base_model.ontology.classes(),
    detections=results
)
# label all images in a folder called `context_images`
base_model.label(input_folder="/Users/krishnaiyer/generative-ai-agentic-cv-base/data/processed/DAWN/Fog", extension=".jpg",human_in_the_loop=True)