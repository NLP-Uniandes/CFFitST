import torch
from sentence_transformers import SentenceTransformer, models, losses
from itertools import product
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
import os
import numpy as np
from numpy.linalg import norm
import json

class CFFitST():
    """
    Custom class for Sentence Transformer few-shot fitting.

    Attributes:
        st_model (SentenceTransformer): Instance of SentenceTransformer.
        size_embedding (int): Size of the embedding vector.
        dic_trazability (dict): Dictionary for traceability information.
        labels (list): List to store labels.
        device (str): Computation device, 'cuda' or 'cpu'.
    """
    def __init__(self, st_model):
        """
        Initializes the CFFitST class with a SentenceTransformer model.

        Args:
            st_model (SentenceTransformer): A SentenceTransformer model instance.
        """
        self.st_model = st_model
        # Determine embedding size from a test encoding
        self.size_embedding = st_model.encode("test").shape[0]
        self.dic_trazability = {} # Dictionary for traceability information
        self.labels = [] # List to store labels
        
        # Set device based on CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.st_model.to(self.device)
    
    def from_pretrained(model_pretrained: str):
        """
        Creates an instance of CFFitST from a pretrained model.

        Args:
            model_pretrained (str): The path or identifier of the pretrained model.

        Returns:
            CFFitST: An instance of the CFFitST class.
        """
        return CFFitST(SentenceTransformer(model_pretrained))
    
    def to(self, device: str):
        """
        Sets the device for computation.

        Args:
            device (str): The computation device to use ('cuda' or 'cpu').
        """
        self.device = device
        self.st_model.to(device)
    
    def set_trazability_info(self, dic:dict):
        """
        Sets traceability information.

        Args:
            dic (dict): A dictionary containing traceability information.
        """
        self.dic_trazability["info"] = dic


    def encode(self, input_st):
        """
        Encodes a single string or a batch of strings.

        Args:
            input_st (str or list): Input text or list of texts to be encoded.

        Returns:
            numpy.ndarray: Encoded embeddings of the input text(s).
        """
        if isinstance(input_st,str):
            return self.st_model.encode(input_st, normalize_embeddings=True)
        else:
            return self.encode_batch(input_st)
    
    def encode_batch(self, list_text):
        """
        Encodes a batch of texts.

        Args:
            list_text (list): A list of texts to be encoded.

        Returns:
            numpy.ndarray: Encoded embeddings of the input texts.
        """
        ten = np.zeros((len(list_text),self.size_embedding))
        for i, text in enumerate(list_text):
            ten[i] = self.st_model.encode(text, normalize_embeddings=True)
        return ten
    
    def fit(self, df, labels, random_state=23, column_label="label", column_sentence="text",\
            epochs=2, validation_data=0.02, chunk_size=0.2,\
            positive_threshold=0.5, negative_threshold=0.5,\
            chunks_reviewed =1, batch_size = 32, min_chunk_size = 0, verbose=True,\
            save_path = "embeddings", name="model", fixed_sizes=False, learning_rate=0.001):
        """
        Fits the Sentence Transformer model to the given data.

        Args:
            df (pd.DataFrame): Dataframe containing the data.
            labels (list): List of labels.
            random_state (int): Random state for reproducibility.
            column_label (str): Name of the column containing labels.
            column_sentence (str): Name of the column containing text.
            epochs (int): Number of epochs for training.
            validation_data (float): Proportion of data used for validation.
            chunk_size (float): Size of chunks for processing.
            positive_threshold (float): Threshold for positive similarity.
            negative_threshold (float): Threshold for negative similarity.
            chunks_reviewed (int): Number of chunks to be reviewed.
            batch_size (int): Size of the batch for training.
            min_chunk_size (int): Minimum size of a chunk.
            verbose (bool): Verbosity of training process.
            save_path (str): Path to save the model.
            name (str): Name of the model.
            fixed_sizes (bool): Whether to use fixed sizes for chunks.
            learning_rate (float): Learning rate used in learning algorithm

        Returns:
            SentenceTransformer: The trained Sentence Transformer model.
        """

        # Establish default sizes when fixed sizes
        if fixed_sizes:
            if chunk_size < 1:
                chunk_size = 500
            if min_chunk_size < 1:
                min_chunk_size = 100

        
        self.labels = labels
        
        # Find unique label pairs and group them into tuples (a,b,c) -> (ab, ac, bc)
        pairs = [ ]
    
        for i in labels:
            for j in labels:
                if ( (i,j) not in pairs ) and ( (j,i) not in pairs ) and i!=j:
                    pairs.append((i,j))
                    
        
        # Dictionary with dataframes of examples
        examples = {}

        # Generate negative examples between different labels
        for label_a, label_b in pairs:
            # n_a * n_b number of possible combinations
            combinations = list(product(df[df["label"]==label_a][column_sentence], df[df["label"]==label_b][column_sentence]))
            result = pd.DataFrame(combinations, columns=[label_a, label_b])
            # Randomize to not be in dataset order (random state for reproducibility)
            result = result.sample(frac=1, random_state = random_state).reset_index(drop=True)
            examples[(label_a,label_b)] = result

        # Generate positive examples between sentences of the same class
        for label in labels:
            pairs.append((label,label))
            # If not equal, it's n * n - n
            combinations = [(a, b) for a, b in product(df[df["label"]==label][column_sentence], df[df["label"]==label][column_sentence]) if a != b]
            result = pd.DataFrame(combinations, columns=[label+"_1", label+"_2"])
            # Randomize to not be in dataset order (random state for reproducibility)
            result = result.sample(frac=1, random_state = random_state).reset_index(drop=True)
            examples[(label,label)] = result

        self.dic_trazability["config"] = { "save_path": save_path, "labels":labels, "epochs":epochs, "validation_data":validation_data,\
                               "chunk_size":chunk_size, "positive_threshold": positive_threshold, "negative_threshold": negative_threshold,\
                               "chunks_reviewed":chunks_reviewed, "batch_size" : batch_size, "min_chunk_size" : min_chunk_size }
        
        
        def get_dataloader(df):
            """
            Create a dataloader from the given dataframe.

            Args:
                df (DataFrame): The dataframe to create a dataloader from.

            Returns:
                DataLoader: A DataLoader object for the input examples.
            """
            examples = [ InputExample(texts=[str(data["l1"]), str(data["l2"])], label=float(data["score"])) for i, data in df.iterrows() ]
            # mantaining original random seed randomness
            return DataLoader(examples, shuffle=False, batch_size=batch_size)
        
        def validate_data():
            """
            Validate the model using the validation data.

            Returns:
                tuple: A tuple containing the dictionary of accuracies for each label pair, the general accuracy, the sum of accuracies, and a traceability dictionary.
            """
            dic_accuracy = {}
            sum_accuracy = 0
            pct_total = 0
            dic_traz = {}
            
            for labels in pair_info:
                cols = pair_info[labels]["val_data"].columns
                total = pair_info[labels]["val_data"].shape[0]
                correctas = 0
                label = pair_info[labels]["score"]
                size = pair_info[labels]["val_data"].shape[0]
                print("Validate data ",labels,"l",label,"s:",size,"\n[",end="")
                
                for i, row in pair_info[labels]["val_data"].iterrows():
                    embedd_1 = self.st_model.encode(str(row[cols[0]]))
                    embedd_2 = self.st_model.encode(str(row[cols[1]]))
                    cos_sim = np.dot(embedd_1, embedd_2)/(norm(embedd_1)*norm(embedd_2))
                    
                    if(label == 0):
                        #print(label,cos_sim,"<=",negative_threshold)
                        if(cos_sim<=negative_threshold):
                            correctas+=1
                    elif(label == 1):
                        #print(label,cos_sim,">=",positive_threshold)
                        if(cos_sim>=positive_threshold):
                            correctas+=1
                    if( i % int(np.floor(size/10)) == 0):
                        print("=",end="")
                dic_accuracy[(labels[0],labels[1])] = correctas/total
                dic_traz[str(labels[0]+"_"+labels[1])] = correctas/total
                sum_accuracy += correctas/total
                pct_total += 1 - correctas/total
                print("]",correctas,total,correctas/total)
                
            print("accuracy",dic_accuracy)
            return dic_accuracy, sum_accuracy/len(pair_info.keys()), sum_accuracy, dic_traz
            
        # Pair information
        pair_info = {}
        

        # Initialization of all label pairs
        for label_a, label_b in pairs:
            data = examples[label_a, label_b]
            # Total number of rows
            size_total = data.shape[0]
            # Validation data
            val_data = data[0:int(np.floor(size_total*validation_data))]
            # Training data
            train_data = data[int(np.floor(size_total*validation_data)):]
            # Score for cosine similarity
            score = 0
            if( label_a == label_b):
                score = 1
            # Save information in the dictionary
            pair_info[(label_a,label_b)] = {"size_total" : size_total, "train_data": train_data, "val_data": val_data, "score" : score, "train_size": train_data.shape[0]}
        


        optimizer = torch.optim.Adam(params=self.st_model.parameters(), lr=0.0001)
        # Loss function is cosine similarity
        train_loss = losses.CosineSimilarityLoss(self.st_model)

        #--------------------------------------------------
        # Main training
        
        index_chunk = 0
        
        # To keep the best one, but not keep everything in memory
        accuracy_memory = []

        last_iteration_space = False


        # Initialization of accuracy for weights
        dic_accuracy = {}
        for i,j in pairs:
            dic_accuracy[i,j] = 0

        # Initialization of starting positions
        dic_end_position = {}
        for i,j in pairs:
            dic_end_position[i,j] = 0
        sum_accuracy = 0

        # Total PCT available
        n_pairs = len(pairs)
        total_pct = n_pairs * chunk_size

        reached_maximum = False
        out_now = False

        while index_chunk < chunks_reviewed and not reached_maximum:

            print(self.dic_trazability)

             # Acquire the data and concatenate it
            df_train = pd.DataFrame()
            df_train["l1"] = []
            df_train["l2"] = []
            df_train["score"] = []
            
            # Generation of dataloader
            print("Iteration",index_chunk)
            
            df_train = pd.DataFrame()
            
            for label_a, label_b in pair_info:
                # The start is the ending of the other
                start_position = dic_end_position[label_a,label_b]
                # MODE 1: Eager variable size assignment
                if fixed_sizes:
                    if index_chunk == 0:
                        size = ( chunk_size / len(self.labels) )
                        end_position = start_position + ( chunk_size / len(self.labels) )
                    else: 
                        pct_total_inv =  n_pairs - sum_accuracy
                        if pct_total_inv == 0:
                            chunk_adaptative_size = (1/n_pairs) * chunk_size
                            size = min_chunk_size + chunk_adaptative_size
                        else:
                            chunk_adaptative_size = ( (1 - dic_accuracy[label_a,label_b])/pct_total_inv ) * chunk_size
                            size = min_chunk_size + chunk_adaptative_size
                            
                        end_position = start_position + size

                        if end_position >  pair_info[label_a, label_b]["size_total"]:
                            #out_now = True
                            reached_maximum = True
                            end_position = pair_info[label_a, label_b]["size_total"]
                            print("max_reached")
                # MODE 2: Conservative Size Assignment
                else:
                    # sum_acuracy to distribute variable size
                    if (n_pairs - sum_accuracy) == 0:
                        chunk_adaptative_size = chunk_size / len(pairs)
                    else:
                        chunk_adaptative_size =(  (1 - dic_accuracy[label_a,label_b]) / (n_pairs - sum_accuracy) )  * chunk_size
                    size = ( min_chunk_size + chunk_adaptative_size ) * pair_info[label_a, label_b]["size_total"]
                    if start_position + size >  pair_info[label_a, label_b]["size_total"]:
                        end_position = pair_info[label_a, label_b]["size_total"]
                        print("max_reached")
                        reached_maximum = True
                    else:
                        end_position = start_position + size

                
                dic_end_position[label_a,label_b] = end_position
                add = pair_info[label_a, label_b]["train_data"][int(start_position):int(end_position)]
                add.columns = ["l1","l2"]
                add["score"] = pair_info[(label_a, label_b)]["score"]
                df_train = pd.concat([df_train, add],axis=0)
                
                
                print(label_a,label_b,"st_pos",start_position,"size",size,"end_pos",end_position)

            if out_now:
                print("OUT_NOW",index_chunk)
                break
            
            # Random order and data loading
            df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            # Fill null values
            df_train["l1"].fillna("")
            df_train["l2"].fillna("")
            train_dataloader = get_dataloader(df_train)

            # Training with SentenceTransformer method using customized examples
            self.st_model.fit(train_objectives=[(train_dataloader, train_loss)], 
                               epochs=epochs,
                               show_progress_bar=verbose)
            
            # Model evaluation
            dic_accuracy, gen_accuracy, sum_accuracy, dic_traz = validate_data()
            dic_new_chunks = {}
            dic_new_pos = {}
        
            self.dic_trazability[index_chunk] = dic_traz
            
            print("new_pos",dic_new_pos,"\n") 
            
            self.st_model.save("iterations/"+name+"/"+str(index_chunk)+".model")
            
            accuracy_memory.append(gen_accuracy)
            
            index_chunk += 1 
            
            print(self.dic_trazability)
            with open("iterations/"+name+"/trazability.json", 'w') as fp:
                json.dump(self.dic_trazability, fp)
        
        # Improving if 2 or more max values are equal, take the last
        #i_mejor = np.argmax(accuracy_memory)   
        i_mejor = len(accuracy_memory)-1
        val_accuracy_mejor = 0
        for i in  reversed(range(len(accuracy_memory))):
            if accuracy_memory[i] > val_accuracy_mejor:
                i_mejor = i
                val_accuracy_mejor = accuracy_memory[i]
        
        self.st_model = SentenceTransformer("iterations/"+name+"/"+str(i_mejor)+".model")
        self.st_model.save(save_path+"/"+name+".model")
        self.dic_trazability[index_chunk] = dic_traz

        with open(save_path+"/"+name+".model/trazability.json", 'w') as fp:
                json.dump(self.dic_trazability, fp)
        

        return self.st_model   

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class ClassificationHead(nn.Module):
    """
    A classification head for the Sentence Transformer model.

    Attributes:
        cff_model (CFFitST): An instance of CFFitST class.
        n_labels (int): Number of labels in the classification task.
        fc (nn.Linear): Fully connected layer.
        softmax (nn.Softmax): Softmax layer.
    """
    def __init__(self, cff_model):
        """
        Initializes the ClassificationHead with a CFFitST model.

        Args:
            cff_model (CFFitST): An instance of CFFitST.
        """
        super(ClassificationHead, self).__init__()
        self.cff_model = cff_model
        self.n_labels = len(cff_model.labels)
        self.fc = nn.Linear(self.cff_model.size_embedding, self.n_labels)
        #self.act = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass for the classification head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after classification.
        """
        x = self.cff_model.encode(x)
        
        x = torch.tensor(x, dtype=torch.float32).to(self.cff_model.device)
        #print(self.fc.weight.data[0].dtype, x.dtype)
        x = self.fc(x)
        #x = self.act(x)
        output = self.softmax(x)
        return output

    def predict(self, x):
        """
        Predicts the class labels for the input data.

        Args:
            x (list or str): Input text or list of texts.

        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.forward(x).cpu().detach().numpy()

    
    def fit(self,x, y, epochs=20, batch_size=16):
        """
        Fits the classification head to the provided data.

        Args:
            x (list): List of input texts.
            y (list): List of target labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.

        Returns:
            None
        """
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        y = torch.tensor(y).to(self.cff_model.device)
        batches_per_epoch = len(x) // batch_size
 
        for epoch in range(epochs):
            with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0, position=0, leave=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for i in bar:
                    # take a batch
                    start = i * batch_size
                    X_batch = x[start:start+batch_size]
                    y_batch = y[start:start+batch_size]
                    # forward pass
                    y_pred = self(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()