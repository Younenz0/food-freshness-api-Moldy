import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import models, transforms

app = FastAPI()

# Class names in the same order as during training
CLASS_NAMES = ['fresh', 'rotten']
FRESH_CERTAINTY_THRESHOLD = 0.95  # 95% for confident fresh
ROTTEN_CERTAINTY_THRESHOLD = 0.7  # 70% for confident rotten
FRESH_LEANING_MIN = 0.85  # 85% minimum for leaning fresh
ROTTEN_LEANING_MIN = 0.5  # 50% minimum for leaning rotten
MODEL_PATH = 'model.pth'

# Preprocessing pipeline (should match training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def create_model_from_name(model_name):
    """Create model architecture based on name"""
    if 'efficientnet' in model_name.lower():
        if 'b0' in model_name.lower():
            model = models.efficientnet_b0(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        elif 'b1' in model_name.lower():
            model = models.efficientnet_b1(weights=None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif 'resnet' in model_name.lower():
        if '18' in model_name.lower():
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
        elif '50' in model_name.lower():
            model = models.resnet50(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    elif 'densenet' in model_name.lower():
        if '121' in model_name.lower():
            model = models.densenet121(weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    else:
        # Default fallback
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    return model

def detect_model_architecture(state_dict):
    """Automatically detect model architecture from state dict keys"""
    state_dict_keys = list(state_dict.keys())
    
    # Check for EfficientNet
    if any('classifier.1.weight' in key for key in state_dict_keys):
        return 'efficientnet_b0'
    
    # Check for ResNet
    elif any('fc.weight' in key for key in state_dict_keys):
        if any('layer4.1.conv2.weight' in key for key in state_dict_keys):
            return 'resnet50'
        else:
            return 'resnet18'
    
    # Check for DenseNet
    elif any('classifier.weight' in key for key in state_dict_keys) and any('denseblock' in key for key in state_dict_keys):
        return 'densenet121'
    
    # Default fallback
    return 'resnet18'

# Load model at startup
def load_model():
    # Load the state dict first to detect architecture
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Detect architecture automatically
    model_type = detect_model_architecture(state_dict)
    print(f"Detected model architecture: {model_type}")
    
    # Create the appropriate model
    model = create_model_from_name(model_type)
    
    # Load the weights
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, 1)
            prob, pred = torch.max(probs, 1)
            prob = prob.item()
            pred = pred.item()
            
            # Get probabilities for both classes
            fresh_prob = probs[0][0].item()
            rotten_prob = probs[0][1].item()
            
            # New logic for classification with leaning thresholds
            print(f"DEBUG: fresh_prob={fresh_prob:.4f}, rotten_prob={rotten_prob:.4f}")
            print(f"DEBUG: FRESH_CERTAINTY_THRESHOLD={FRESH_CERTAINTY_THRESHOLD}, ROTTEN_CERTAINTY_THRESHOLD={ROTTEN_CERTAINTY_THRESHOLD}")
            print(f"DEBUG: FRESH_LEANING_MIN={FRESH_LEANING_MIN}, ROTTEN_LEANING_MIN={ROTTEN_LEANING_MIN}")
            
            if fresh_prob >= FRESH_CERTAINTY_THRESHOLD:
                # Very confident fresh
                label = "fresh"
                leaning = None
                print(f"DEBUG: Chose FRESH (confident)")
            elif rotten_prob >= ROTTEN_CERTAINTY_THRESHOLD:
                # Very confident rotten
                label = "rotten"
                leaning = None
                print(f"DEBUG: Chose ROTTEN (confident)")
            elif fresh_prob >= FRESH_LEANING_MIN:
                # Leaning fresh (85-95%)
                label = "uncertain"
                leaning = "fresh"
                print(f"DEBUG: Chose LEANING FRESH")
            elif rotten_prob >= ROTTEN_LEANING_MIN:
                # Leaning rotten (50-70%)
                label = "uncertain"
                leaning = "rotten"
                print(f"DEBUG: Chose LEANING ROTTEN")
            else:
                # Very uncertain (below 50% for both)
                label = "uncertain"
                leaning = "unknown"
                print(f"DEBUG: Chose UNKNOWN")
            
            return JSONResponse({
                "label": label, 
                "certainty": round(prob, 4),
                "leaning": leaning,
                "fresh_probability": round(fresh_prob, 4),
                "rotten_probability": round(rotten_prob, 4)
            })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)