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
    def __init__(self, st_model):
        self.st_model = st_model
        self.size_embedding = st_model.encode("xd").shape[0]
        self.dic_trazability = {}
        self.labels = []
    
    def from_pretrained(model_pretrained: str):
        return CFFitST(SentenceTransformer(model_pretrained))
    
    def to(self, device: str):
        self.device = device
        self.st_model.to(device)
    
    def set_trazability_info(self, dic:dict):
        self.dic_trazability["info"] = dic

    def encode(self, input_st):
        if isinstance(input_st,str):
            return self.st_model.encode(input_st, normalize_embeddings=True)
        else:
            return self.encode_batch(input_st)
    
    def encode_batch(self, list_text):
        ten = np.zeros((len(list_text),self.size_embedding))
        for i, text in enumerate(list_text):
            ten[i] = self.st_model.encode(text, normalize_embeddings=True)
        return ten
    
    def fit(self, df, labels, random_state=23, column_label="label", column_sentence="text",\
            epochs=2, validation_data=0.02, chunk_size=0.2,\
            positive_threshold=0.5, negative_threshold=0.5,\
            chunks_reviewed =1, batch_size = 32, min_chunk_size = 0, verbose=True,\
            save_path = "embeddings", name="model"):
        
        self.labels = labels
        
        # Find unique label pairs and group them into tuples (a,b,c) -> (ab, ac, bc)
        pairs = [ pairs.append((i,j)) for i in labels for j in labels if ( (i,j) not in pairs ) and ( (j,i) not in pairs ) and i!=j ]
        
        #REFACTORED
        #for i in labels:
        #    for j in labels:
        #        if ( (i,j) not in pairs ) and ( (j,i) not in pairs ) and i!=j:
                    
        
        # Diccionario con dfs de ejemplos
        ejemplos = {}

        # Genera ejemplos (negativos) entre los diferentes
        for label_a, label_b in pairs:
            # n_a * n_b número de combinaciones posibles
            combinations = list(product(df[df["label"]==label_a][column_sentence], df[df["label"]==label_b][column_sentence]))
            result = pd.DataFrame(combinations, columns=[label_a, label_b])
            # aleatoriedad para que no esten en el orden del dataset (random state para reproducción)
            result = result.sample(frac=1, random_state = random_state).reset_index(drop=True)
            ejemplos[(label_a,label_b)] = result

        # Genera ejemplos (positivos) entre sentencias de la misma clase
        for label in labels:
            # Si no son iguales n * n - n
            combinations = [(a, b) for a, b in product(df[df["label"]==label][column_sentence], df[df["label"]==label][column_sentence]) if a != b]
            result = pd.DataFrame(combinations, columns=[label+"_1", label+"_2"])
            result = result.sample(frac=1, random_state = random_state).reset_index(drop=True)
            ejemplos[(label,label)] = result

        self.trazability_dic["config"] = { "save_path": save_path, "labels":labels, "epochs":epochs, "validation_data":validation_data,\
                               "chunk_size":chunk_size, "positive_threshold": positive_threshold, "negative_threshold": negative_threshold,\
                               "chunks_reviewed":chunks_reviewed, "batch_size" : batch_size, "min_chunk_size" : min_chunk_size }

        # CODIGO ORIGINARIO DE FINED TUNED
        def enough_room():
            res = True
            for label_1, label_2 in pair_info:
                if pair_info[(label_1,label_2)]["adaptive_starting_index"] > pair_info[(label_1,label_2)]["train_size"]-1:
                    res = False
            return res
        
        def get_dataloader(df):
            ejemplos = [ InputExample(texts=[str(data["l1"]), str(data["l2"])], label=float(data["score"])) for i, data in df.iterrows() ]
            # mantaining original random seed randomness
            return DataLoader(ejemplos, shuffle=False, batch_size=batch_size)
        
        def validate_data():
            dic_inv_norm_accuracy = {}
            dic_accuracy = {}
            accuracy_sum = 0
            pct_total = 0
            for labels in pair_info:
                cols = pair_info[labels]["val_data"].columns
                total = pair_info[labels]["val_data"].shape[0]
                correctas = 0
                label = pair_info[labels]["score"]
                size = pair_info[labels]["val_data"].shape[0]
                print("Validate data ",labels,"l",label,"s:",size,"\n[",end="")
                for i, row in pair_info[labels]["val_data"].iterrows():
                    embedd_1 = self.st_model.encode(str(row[cols[1]]))
                    embedd_2 = self.st_model.encode(str(row[cols[2]]))
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
                dic_inv_norm_accuracy[labels] = 1 - correctas/total
                dic_accuracy[labels[0]+"_"+labels[1]] = correctas/total
                accuracy_sum += correctas/total
                pct_total += 1 - correctas/total
                print("]",correctas,total,correctas/total)
            
            for labels in pair_info:
                if pct_total !=0:
                    dic_inv_norm_accuracy[labels] = (dic_inv_norm_accuracy[labels])/pct_total
                else:
                    dic_inv_norm_accuracy[labels] = 1/len(list(pair_info.keys()))
                
            print("accuracy",dic_accuracy)
            return dic_inv_norm_accuracy, accuracy_sum/len(pair_info.keys()), dic_accuracy
            
        # Información de pares
        pair_info = {}
        initial_chunk_size_repo = np.floor(chunk_size/len(labels))

        # Inicialización de todos las parejas de labels
        for label_a, label_b in pairs:
            data = ejemplos[label_a, label_b]
            # cantidad de filas
            size_total = data.shape[0]
            # datos de validacion
            val_data = data[0:int(np.floor(size_total*validation_data))]
            # datos de entrenamiento
            train_data = data[int(np.floor(size_total*validation_data)):]
            # score para similitud coseno
            score = 0
            if( label_a == label_b):
                score = 1
            # guardar información en el diccionario
            pair_info[(label_a,label_b)] = {"size_total" : size_total, "train_data": train_data, "val_data": val_data, "score" : score, "train_size": train_data.shape[0], "chunk_adaptive_size": initial_chunk_size_repo, "adaptive_starting_index": 0}
        


        optimizer = torch.optim.Adam(params=self.st_model.parameters(), lr=0.0001)
        # Función de error es seno y coseno
        train_loss = losses.CosineSimilarityLoss(self.st_model)

        # Entrenamiento principal
        index_chunk = 0
        
        # Para quedarse con el mejor, pero no mantener todo en memoria
        accuracy_memory = []

        last_iteration_space = False

        while index_chunk < chunks_reviewed and enough_room:

            print(self.trazability_dic)

            # adquirir los datos y concatenarlos
            df_train = pd.DataFrame()
            df_train["l1"] = []
            df_train["l2"] = []
            df_train["score"] = []
            
            # generación de datoloade
            for label_a, label_b in pair_info:
                # posicion inicio adaptaive_starting_index seteado arriba
                start_position = int(pair_info[(label_a, label_b)]["adaptive_starting_index"])
                # posicion final posicion inicial + tamano variable + tamano minimo chunk
                end_position = int(pair_info[(label_a, label_b)]["adaptive_starting_index"] + pair_info[(label_a, label_b)]["chunk_adaptive_size"]) + min_chunk_size
                # sacamos el train position
                df_label_label = pair_info[(label_a, label_b)]["train_data"][start_position:end_position]
                # renombramos para simplificar
                df_label_label.columns = ["index","l1","l2"]
                # se pone para una clase siempre el mismo score
                df_label_label["score"] = pair_info[(label_a, label_b)]["score"]
                # data frame con los ejemplos
                df_train = pd.concat([df_train,df_label_label])
            
            # random order y carga de datos
            df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            # llenar los nulos
            df_train["l1"].fillna("")
            df_train["l2"].fillna("")
            train_dataloader = get_dataloader(df_train)

            # entrenamiento
            self.st_model.fit(train_objectives=[(train_dataloader, train_loss)], 
                               epochs=epochs,
                               show_progress_bar=verbose)
            
            # evaluacion de modelo
            dic, gen_accuracy, dic_accuracy = validate_data()
            dic_new_chunks = {}
            dic_new_pos = {}

            for label_a, label_b in dic:
                new_position = int(pair_info[(label_a, label_b)]["adaptive_starting_index"] + pair_info[(label_a, label_b)]["chunk_adaptive_size"])
                pair_info[(label_a, label_b)]["adaptive_starting_index"] = new_position
                dic_new_pos[(label_a, label_b)] = new_position
                new_chunk = int(np.floor(dic[(label_a,label_b)]*chunk_size*pair_info[(label_a, label_b)]["size_total"])) 
                # se agrega el min chunk_size
                pair_info[(label_a, label_b)]["chunk_adaptive_size"] = new_chunk 
                dic_new_chunks[(label_a, label_b)] = pair_info[(label_a, label_b)]["chunk_adaptive_size"]

            print("new_chunks",dic_new_chunks,"\n")
        
            self.trazability_dic[index_chunk] = dic_accuracy
            
            print("new_pos",dic_new_pos,"\n") 
            
            self.st_model.save("iterations/"+name+"/"+str(self.items)+"/"+str(index_chunk)+".model")
            
            accuracy_memory.append(gen_accuracy)
            
            index_chunk += 1 
            
            
            with open("iterations/"+name+"/trazability.json", 'w') as fp:
                json.dump(self.trazability_dic, fp)
        
        
        i_mejor = np.argmax(accuracy_memory)
        
        self.st_model = SentenceTransformer("iterations/"+name+"/"+str(i_mejor)+".model")
        self.st_model.save(save_path+"/"+name+".model")
        self.trazability_dic[index_chunk] = dic_accuracy

        with open(save_path+"/"+name+".model/trazability", 'w') as fp:
                json.dump(self.trazability_dic, fp)
        

        return self.st_model   

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class ClassificationHead(nn.Module):

    def __init__(self, cff_model):
        super(ClassificationHead, self).__init__()
        self.cff_model = cff_model
        self.n_labels = len(cff_model.labes)
        self.fc = nn.Linear(self.cff_model.size_embedding, output_size=self.cff_model.n_labels)
        self.act = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
    

    def forward(self, x):
        x = self.cff_model.encode(x)
        x = self.fc(x)
        x = self.act(x)
        output = self.softmax(x, dim=1)
        return output

    
    def fit(self,x, y, n_epochs=20, batch_size=16):



        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        batches_per_epoch = len(x) // batch_size
 
        for epoch in range(n_epochs):
            with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
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






    

    



        
        
    

        

