#!/usr/bin/env python
# coding: utf-8

# In[70]:


from xyz import Restaurant_Name_generator
import os
os.environ['OPENAI_API_KEY']=Restaurant_Name_generator


# In[71]:


from langchain.llms import OpenAI
llm=OpenAI(temperature=0.6)
ans=llm("I want to open a resturant for Indian food.Suggest a fancy name for this")
print(ans)


# In[57]:


import sys
print(sys.executable)


# In[60]:


from langchain.prompts import PromptTemplate
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a resturant for {cuisine} food.Suggest a fancy name for this"
    
)
prompt_template_name.format(cuisine="Italian")


# In[62]:


from langchain.chains import LLMChain
chain=LLMChain(llm=llm,prompt=prompt_template_name)
chain.run("American")


# In[76]:


llm=OpenAI(temperature=0.6)
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a resturant for {cuisine} food.Suggest a fancy name for this"
    
)
name_chain=LLMChain(llm=llm,prompt=prompt_template_name)
prompt_template_items=PromptTemplate(
    input_variables=['restaurant_name'],
    template="""Suggest some food menu items for{restaurant_name}.Return it in comma separated string"""
    
)
food_items_chain=LLMChain(llm=llm,prompt=prompt_template_items)


# In[78]:


from langchain.chains import SimpleSequentialChain
chain=SimpleSequentialChain(chains=[name_chain,food_items_chain])
response=chain.run("Indian")
print(response)


# In[80]:


llm=OpenAI(temperature=0.6)
prompt_template_name=PromptTemplate(
    input_variables=['cuisine'],
    template="I want to open a resturant for {cuisine} food.Suggest a fancy name for this"
    
)
name_chain=LLMChain(llm=llm,prompt=prompt_template_name,output_key="restaurant_name")
prompt_template_items=PromptTemplate(
    input_variables=['restaurant_name'],
    template="""Suggest some food menu items for{restaurant_name}.Return it in comma separated string"""
    
)
food_items_chain=LLMChain(llm=llm,prompt=prompt_template_items,output_key="menu_items")


# In[90]:


from langchain.chains import SequentialChain
chain= SequentialChain(
    chains=[name_chain,food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name','menu_items']
)
result=chain({'cuisine':'Indian'})
print(result)


# In[ ]:




