#!/usr/bin/env python
# coding: utf-8

# # WORKSHOP INTRODUÇÃO À MACHINE LEARNING
# Baseado no tutorial :
# Autoria: Lilianne M. I. Nakazono | Organização: Pyladies SP | Local: FIAP Paulista
# 
# Esse tutoria contém  minhas anotações do Workhop
# 

'''

# Importando bibliotecas:
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import metrics


# In[101]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    print (cm)
    return(cm)
    


# In[5]:


'''
3. Leitura de dados
'''
#  Star/Galaxy dataset

url = 'https://raw.githubusercontent.com/marixko/'
file = 'tutorial_classifiers/master/tutorial_data.txt'
data = pd.read_csv(url+file, delim_whitespace=True, low_memory=False)


# In[8]:


len(data.columns)


# In[6]:


data.columns


# In[7]:


data.iloc[0:20, 20]   # índice 20 é a coluna 21 = Classe do objeto ( estrela/galaxia)


# In[16]:


'''
'FWHM_n', 'A', 'B', 'KrRadDet',
       'uJAVA_auto', 'F378_auto', 'F395_auto', 'F410_auto', 'F430_auto',
       'g_auto', 'F515_auto', 'r_auto', 'F660_auto', 'i_auto', 'F861_auto',
       'z_auto',
'''

data2=data.iloc[:,4:22]   #ou [ 4 ; len(data.columns)+1] 
# DEtalhe: nao existe coluna c/ indice 22, mas comando iloc interpreta que queremos até a última coluna


# In[17]:


len(data2.columns)


# In[12]:


data2.iloc[0:20,0]


# In[18]:


data2.columns


# In[20]:


'''
4.  ANALISE EXPLORATORIA  DE DADOS
'''

data2.corr() 


# In[26]:


fig,ax=plt.subplots(figsize=(15,15))

chart=sns.heatmap( data2.corr(), cmap='bwr', center=0, square=True, annot=True)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!


# In[38]:


atribut2=data2.iloc[:,0:16] #até a coluna 15  (o 16 nao entra)

class2=data2.iloc[:,16]
#len(class2.columns)  # -----NAO FUNCIONOU
class2   # 'E o y real QUE QUEREMOS estimar '


# In[39]:


'''
5. Amostragem

Debe-se separar 
Para comparar a performance de diversos modelos (por 
exemplo, com diferentes parâmetros), é importante 
adicionar mais uma etapa: a validação 

'''

#Amostragem para validação cruzada

#target: tem informaçòes das classes
x_train, x_test, y_train, y_test = train_test_split(atribut2, 
                                                    class2,
                                                    test_size=0.3, 
                                                    random_state=2)   ## regra "randomica"

#amostragem aleatorio: no micro tudo é pseu-daleatorio.
#random state é  um numero que sempre as mesmas amostras de teste e treinamento.
#  sempre tera a mesma amostra qd random_state=2


# In[42]:


x_train[0:10]


# In[156]:


''' 
6. Geração de modelo e VALIDACAO


svm.SVC()
tree.DecisionTreeClassifier()
ensemble.RandomForestClassifier()
neighbours.KNeighborsClassifier()

'''


# In[157]:



'''
MODELO SVC

A separação não precisa ser necessariamente linear 
Para tanto, modifica-se o kernel
'''
clf2 = SVC(kernel='linear')
#clf2.fit(x_train,y_train)
#ou
clf2.fit(x_train,y_train.values.ravel())


# In[106]:


y_pred_svc=clf2.predict(x_test)

matrix_svc=confusion_matrix(y_test,y_pred_svc)


# In[159]:


model="  SVC"
#fig,ax=plt.subplots(figsize=(9,9))

fig= plot_confusion_matrix(matrix_svc,classes=['0','1'])
plt.title('Confusion matrix \n model:%s' %model)   #replace original title
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


'''
plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): 
'''


# In[ ]:


'''
Matriz de confusão
                         Predito pelo  modelo
                        Estrela      Galaxia
            Estrela       VP           FN
Verdadeiro  
            Galaxia       FP           VN

      
Métricas de performance

• Acurácia
(VP + VN) / Total
• Precisão (+)
VP / (VP + FP)
• Recall (+)
VP / (VP + FN) 
• F-score
2 (Precision x Recall) / (Precision + Recall)
Como quantificar a performance do seu modelo?  

'''


# In[166]:


# https://www.geeksforgeeks.org/multi-dimensional-lists-in-python/
#result_models=[[0 for x in range(5)] for x in range(m)] 

#result_models=[["Model "," VP  ","  FP ", "VN  ","FN  ","Acuracy  ", "Precision   ", "Recall  ","F-score  "]]
result_models=[["Model","Acuracy", "Precision", "Recall","F-score"]]


# In[167]:


matrix_model=matrix_svc
cmresult=[model]
VP=matrix_model[0][0]
cmresult.append(VP)


# In[168]:



# model="SVC" already created before
cmresult=[model]
VP=matrix_model[0][0]
#cmresult.append(VP)
FP=matrix_model[1][0]
#cmresult.append(FP)
VN=matrix_model[0][1]
#cmresult.append(VN)
FN=matrix_model[1][1]
#cmresult.append(FN)
Acur=(VP+VN)/(VP+VN+FP+FN)
cmresult.append('%.5f' %Acur)
Prec=VP / (VP + FP)
cmresult.append('%.5f' % Prec)
Recall=VP / (VP + FN)
cmresult.append('%.5f' %Recall)
F_score=2* (Prec * Recall) / (Prec + Recall)
cmresult.append('%.5f' %F_score)
result_models.append(cmresult) 
result_models


# In[169]:


for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# # PART 2 
# Testing several models
# -------------------------------------------------------

# In[178]:



'''
Como MAIS DE 1 MODELO será  TESTADO:
     svm.SVC()
    tree.DecisionTreeClassifier()
    ensemble.RandomForestClassifier()
    neighbours.KNeighborsClassifier()

Então VAMOS REPETIR A ETAPA DE AMOSTRAGEM
p/ VALIDAÇAO CRUZADA

Entao precisamos divir o dataset. 
Cada modelos irá contar com um dataset de teste e validação diferentes.

            DATASET: [estrelas +galaxias/
                            | (amostragem aleatoria)
                            v
        Treinamento               ValidaçÃO
        [estrelas+galaxias]    [estrelas+galaxias]     
            |
            v  (mais uma etapa de amostragem aleatoria: dividir em (i) partes , para i modelos)

Treinamento(i)               ValidaçÃO
[estrelas+galaxias](i)    [estrelas+galaxias] (i)  

Exemplo: 4 modelos
          Dataset "Treinamento" subdividido
          p/ teste ou validação (V)  
          
                      Amostra
            Modelo 1:   (V)   1   2   3
            Modelo 2:    0   (V)  2   3
            Modelo 3:    0    1  (V)  3
            Modelo 4:    0    1   2  (V)

'''
# 1 fase : fazer 1a  divisão do dataset principal
x_train, x_test, y_train, y_test = train_test_split(atribut2, 
                                                    class2,
                                                    test_size=0.3, 
                                                    random_state=2)   ## regra "randomica"

len(x_train)


# In[180]:


nmodels=4
nsample=int(len(x_train)/nmodels)
x_sample=nmodels*[0]
y_sample=nmodels*[0]
c=0
for i in range(nmodels):
    x_sample[i]=x_train[c:(c+nsample)]
    y_sample[i]=y_train[c:(c+nsample)]
    c+=nsample
    
len(x_sample[nmodels-1])


# In[224]:


print(x_sample[0])  #,x_sample[nmodels-1])


# In[225]:


print(x_sample[nmodels-1])


# In[ ]:





# In[266]:


'''
MODELO 1 - SVC

A separação não precisa ser necessariamente linear 
Para tanto, modifica-se o kernel

'''

frames_x = [x_sample[1], x_sample[2], x_sample[3]]
frames_y = [y_sample[1], y_sample[2], y_sample[3]]
x_train=pd.concat(frames_x)
y_train=pd.concat(frames_y)
len(x_train)
x_test_s=x_sample[0]
y_test_s=y_sample[0]


# In[267]:


#clf2.fit(x_train,y_train)
#ou
clf_svc = SVC(kernel='linear')
clf_svc.fit(x_train,y_train.values.ravel())


# In[269]:


y_pred_1= clf_svc.predict(x_test_s)

matrix_1=confusion_matrix(y_test_s,y_pred_1)


# In[270]:


model=" SVC"
#fig,ax=plt.subplots(figsize=(9,9))

fig= plot_confusion_matrix(matrix_1,classes=['0','1'])
plt.title('Confusion matrix \n model:%s' %model)   #replace original title
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


# In[271]:


# Run only for first model
result_models=[["Model","Acuracy", "Precision", "Recall","F-score"]]


# In[272]:


# RESULTADOS
matrix_model=matrix_svc
cmresult=[model]
VP=matrix_model[0][0]
cmresult.append(VP)

# model="XXXXX" already created before
cmresult=[model]
VP=matrix_model[0][0]
#cmresult.append(VP)
FP=matrix_model[1][0]
#cmresult.append(FP)
VN=matrix_model[0][1]
#cmresult.append(VN)
FN=matrix_model[1][1]
#cmresult.append(FN)
Acur=(VP+VN)/(VP+VN+FP+FN)
cmresult.append('%.5f' %Acur)
Prec=VP / (VP + FP)
cmresult.append('%.5f' % Prec)
Recall=VP / (VP + FN)
cmresult.append('%.5f' %Recall)
F_score=2* (Prec * Recall) / (Prec + Recall)
cmresult.append('%.5f' %F_score)
result_models.append(cmresult) 
result_models
for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# In[ ]:





# In[ ]:





# In[273]:


'''
MODELO 2 - DEcision Tree Classifier

A separação não precisa ser necessariamente linear 
Para tanto, modifica-se o kernel
'''





frames_x = [x_sample[0], x_sample[2], x_sample[3]]   ####### CHANGE
frames_y = [y_sample[0], y_sample[2], y_sample[3]]    ####### CHANGE
x_train=pd.concat(frames_x)
y_train=pd.concat(frames_y)
len(x_train)
x_test_s=x_sample[1]     ####### CHANGE
y_test_s=y_sample[1]      ####### CHANGE

######## Model structure
clf_dtc = DecisionTreeClassifier()    ####### CHANGE
clf_dtc.fit(x_train,y_train.values.ravel()) ####### CHANGE 


# In[274]:


y_pred_2=clf_dtc.predict(x_test_s)   #####change

matrix_2=confusion_matrix(y_test_s,y_pred_2)    ######change

model=" DecTree"     ################change
#fig,ax=plt.subplots(figsize=(9,9))

fig= plot_confusion_matrix(matrix_2,classes=['0','1'])   #########change
plt.title('Confusion matrix \n model:%s' %model)   #replace original title
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


# In[275]:


# RESULTADOS
matrix_model=matrix_2  ############change
cmresult=[model]
VP=matrix_model[0][0]
cmresult.append(VP)

# model="XXXXX" already created before
cmresult=[model]
VP=matrix_model[0][0]
#cmresult.append(VP)
FP=matrix_model[1][0]
#cmresult.append(FP)
VN=matrix_model[0][1]
#cmresult.append(VN)
FN=matrix_model[1][1]
#cmresult.append(FN)
Acur=(VP+VN)/(VP+VN+FP+FN)
cmresult.append('%.5f' %Acur)
Prec=VP / (VP + FP)
cmresult.append('%.5f' % Prec)
Recall=VP / (VP + FN)
cmresult.append('%.5f' %Recall)
F_score=2* (Prec * Recall) / (Prec + Recall)
cmresult.append('%.5f' %F_score)
result_models.append(cmresult) 
result_models
for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# In[ ]:





# In[247]:


'''
MODELO 3 - ensemble.RandomForestClassifier()

A separação não precisa ser necessariamente linear 
Para tanto, modifica-se o kernel
'''










frames_x = [x_sample[0], x_sample[1], x_sample[3]]   ####### CHANGE
frames_y = [y_sample[0], y_sample[1], y_sample[3]]    ####### CHANGE
x_train=pd.concat(frames_x)
y_train=pd.concat(frames_y)
len(x_train)
x_test_s=x_sample[2]     ####### CHANGE
y_test_s=y_sample[2]      ####### CHANGE

######## Model structure
clf_rf = RandomForestClassifier()    ####### CHANGE
clf_rf.fit(x_train,y_train.values.ravel()) ####### CHANGE 


# In[276]:


y_pred_3=clf_rf.predict(x_test_s)   #####change

matrix_3=confusion_matrix(y_test_s,y_pred_3)    ######change

model=" RanFor"     ################change
#fig,ax=plt.subplots(figsize=(9,9))

fig= plot_confusion_matrix(matrix_3,classes=['0','1'])   #########change
plt.title('Confusion matrix \n model:%s' %model)   #replace original title
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


# In[277]:


# RESULTADOS
matrix_model=matrix_3  ############change
cmresult=[model]
VP=matrix_model[0][0]
cmresult.append(VP)

# model="XXXXX" already created before
cmresult=[model]
VP=matrix_model[0][0]
#cmresult.append(VP)
FP=matrix_model[1][0]
#cmresult.append(FP)
VN=matrix_model[0][1]
#cmresult.append(VN)
FN=matrix_model[1][1]
#cmresult.append(FN)
Acur=(VP+VN)/(VP+VN+FP+FN)
cmresult.append('%.5f' %Acur)
Prec=VP / (VP + FP)
cmresult.append('%.5f' % Prec)
Recall=VP / (VP + FN)
cmresult.append('%.5f' %Recall)
F_score=2* (Prec * Recall) / (Prec + Recall)
cmresult.append('%.5f' %F_score)
result_models.append(cmresult) 
result_models
for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# In[ ]:





# In[ ]:





# In[252]:



'''
MODELO 4 - neighbours.KNeighborsClassifier()
'''


frames_x = [x_sample[0], x_sample[1], x_sample[2]]   ####### CHANGE
frames_y = [y_sample[0], y_sample[1], y_sample[2]]    ####### CHANGE
x_train=pd.concat(frames_x)
y_train=pd.concat(frames_y)
len(x_train)
x_test_s=x_sample[3]     ####### CHANGE
y_test_s=y_sample[3]      ####### CHANGE

######## Model structure
clf_knc = KNeighborsClassifier()    ####### CHANGE
clf_knc.fit(x_train,y_train.values.ravel()) ####### CHANGE 


# In[278]:


y_pred_4=clf_knc.predict(x_test_s)   #####change

matrix_4=confusion_matrix(y_test_s,y_pred_4)    ######change

model=" K-nc"     ################change
#fig,ax=plt.subplots(figsize=(9,9))

fig= plot_confusion_matrix(matrix_4,classes=['0','1'])   #########change
plt.title('Confusion matrix \n model:%s' %model)   #replace original title
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


# In[279]:


# RESULTADOS
matrix_model=matrix_4  ############change
cmresult=[model]
VP=matrix_model[0][0]
cmresult.append(VP)

# model="XXXXX" already created before
cmresult=[model]
VP=matrix_model[0][0]
#cmresult.append(VP)
FP=matrix_model[1][0]
#cmresult.append(FP)
VN=matrix_model[0][1]
#cmresult.append(VN)
FN=matrix_model[1][1]
#cmresult.append(FN)
Acur=(VP+VN)/(VP+VN+FP+FN)
cmresult.append('%.5f' %Acur)
Prec=VP / (VP + FP)
cmresult.append('%.5f' % Prec)
Recall=VP / (VP + FN)
cmresult.append('%.5f' %Recall)
F_score=2* (Prec * Recall) / (Prec + Recall)
cmresult.append('%.5f' %F_score)
result_models.append(cmresult) 
result_models
for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# In[263]:


for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# In[280]:


'''
Model chosen :  DEcTree --> model 3

Use original x_test and y_test
'''
model=" DecTree" 
y_pred_final=clf_dtc.predict(x_test)   #####change

matrix_f=confusion_matrix(y_test,y_pred_final)    ######change

    ################change
#fig,ax=plt.subplots(figsize=(9,9))

fig= plot_confusion_matrix(matrix_f,classes=['0','1'])   #########change
plt.title('Confusion matrix \n model:%s' %model)   #replace original title
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show()


# In[281]:


# RESULTADOS
matrix_model=matrix_f  ############change
cmresult=[model+"_f"]
VP=matrix_model[0][0]
cmresult.append(VP)

# model="XXXXX" already created before
cmresult=[model]
VP=matrix_model[0][0]
#cmresult.append(VP)
FP=matrix_model[1][0]
#cmresult.append(FP)
VN=matrix_model[0][1]
#cmresult.append(VN)
FN=matrix_model[1][1]
#cmresult.append(FN)
Acur=(VP+VN)/(VP+VN+FP+FN)
cmresult.append('%.5f' %Acur)
Prec=VP / (VP + FP)
cmresult.append('%.5f' % Prec)
Recall=VP / (VP + FN)
cmresult.append('%.5f' %Recall)
F_score=2* (Prec * Recall) / (Prec + Recall)
cmresult.append('%.5f' %F_score)
result_models.append(cmresult) 
result_models
for i in range(len(result_models)) :  
    for j in range(len(result_models[i])) :  
        print(result_models[i][j], end=" ") 
    print() 


# In[282]:


'''
Results

Rod.1 )  30% data for validation
          random=2      
'''
result_models_rod_1=result_models


# In[ ]:




