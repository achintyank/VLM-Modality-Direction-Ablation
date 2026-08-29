import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor # transformers is the huggingface transformer library
from datasets import load_dataset
import requests, io
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype = "float16", device_map = "cuda"
) # we just loaded the right Qwen model, and changed the data type, and we're using the cuda gpu
model.eval() # as I learned before, puts the model in eval mode

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", max_pixels = 512 * 512) # we created a processor object based on the downloaded model's processor configuration files
# from_pretrained downloads all architecture and weights, creates objects based on that
#row = next(iter(dataset)) SANITY CHECK
#  print(row.keys())     # confirm the real field names
# figure out why you need this!

# load dataset
dataset = load_dataset("tomg-group-umd/pixelprose", split = "train", streaming=True) # we do split = "train" not because we're training but because the data that we need is under the training split, streaming feeds one row at a time instead of downloading all that at once, bc of this we can't do len()
N_EXAMPLES  = 500 # this is the number of examples we will test with
activations = {L: [] for L in list(range(1,29))} # collect for ALL layers
labels = {L: [] for L in list(range(1,29))}
count = 0 # counts how many successful images you process, you want this to become equal to N_EXAMPLES
image_token_id = model.config.image_token_id # finds out what number signals image token

# now we are going to download and retrieve all of the images
for row in dataset:
    try:
      resp = requests.get(row["url"], timeout=10) # downloads from the image, waits 10 sec for server to respond
      image = Image.open(io.BytesIO(resp.content)).convert("RGB") # converts image bytes in file, puts image into object, converts into RGB
    except Exception:
       continue 
    caption = row["vlm_caption"] # see this!, sets caption??
    # image and caption are BOTH OBJECTS, row is a dictionary with url to caption
    # more notes in notebook

    #Qwen is a chatbot, so we are structuring this as a chat prompt
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": caption}, {"type": "text", "text": "What is this image?"},],
    }] # in dict inside list we give role-user, content-(image dict and text dict), one dict in a list because one chat prompt

    # now, we will format it in the chat template, we say no to tokenization because we need strings, not numbers from tokenization for our processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # add_generation_prompt=True is there because we are adding the last part that signals to the model that it's its turn to answer
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda") # tokenizes text + preprocesses image and puts in the image id where image tokens go, return_tensors="pt" gives PyTorch tensors
    # shape of inputs: batch, seq_len
    # now, we are going to actually run it and save internal states
    with torch.no_grad():
        out = model.model(**inputs, output_hidden_states = True) # we just ran the model in non-gradient (non-training) mode, and we are outputting hidden states (like activations right?)

    # now, we are going to label and store the data
    input_ids = inputs["input_ids"][0]
    is_vision = (input_ids == image_token_id) # we are labeling the image tokens is_vision=True based on the image_token_id
    for L in list(range(1,29)): # L being an integer, collect for ALL layers
        h = out.hidden_states[L][0] # gives the activations for each layer, store in L?
        activations[L].append(h.float().cpu().numpy()) #adds activations for that layer to the dictionary, literally into the value part because activations[L], L being the key, accesses the value spot
        labels[L].append(is_vision.cpu().numpy()) #.cpu() moves to CPU memory, then we convert it to numpy array, stores labels in the labels dictionary, adds this array into the value part of key
    # the list that is the value part of the key-value is actually a list of 40 numpy arrays, each representing the activations of one image (we had a 40 image set!)
    count += 1 # for each image
    if count >= N_EXAMPLES:
        break

#and now we finally train the probe
# we run a logistic regression because ???????????
for L in [1,2,3,4,5,6,7,8]: 
    X_train = np.concatenate(activations[L][0:28], axis=0) # concatenates all activation arrays for all images in layer L
    X_test = np.concatenate(activations[L][28:], axis=0)
    y_train = np.concatenate(labels[L][0:28], axis=0) # concatenates all of the labels in the same way
    y_test = np.concatenate(labels[L][28:], axis = 0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # we split the training and test data sets from the given activations and labels (70% for training, 30% for testing)
    clf = LogisticRegression(max_iter=2000) # logistic regression with max 2000 steps (this is the gradient adjusting stuff right?)
    clf.fit(X_train, y_train) # trains the probe
    acc = clf.score(X_test, y_test) # data type? this is the testing score
    print(f"layer {L}: test accuracy = {acc:.3f}")
    
for L in [11, 16, 22, 28]:
    X_test = np.concatenate(activations[L][28:], axis=0)
    y_test = np.concatenate(labels[L][28:], axis = 0)
    accs = clf.score(X_test, y_test) # testing linear probe with layers 9-28 activations, images 29 to 40
    print(f"layer {L}: test accuracy = {accs:.3f}")
