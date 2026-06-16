import torch 
from .cache_manager import cache_manager
from .llm import model, tokenizer
from .knowledge_loader import load_knowledge
from transformers.cache_utils import DynamicCache



def preload_knowledge(knowledge: str):
    """
    This is the preloading function which will load the knowledge into the cache and then we will use the cache to answer the user query
    Args:
        knowledge (str): The knowledge to be preloaded into the cache
    returns:
        cache (DynamicCache): The cache containing the preloaded knowledge    
    """
    
    cache=DynamicCache()
    inputs=tokenizer(knowledge , return_tensors="pt")
    
    with torch.no_grad():
        outputs=model(**inputs , use_cache=True , past_key_values=cache)
    cache_manager.set_cache(cache)
    return cache    


def answer_question(question: str):
    """ 
    this is the anqwer function which will take the user query and then we will use the cache to answer the user query
    Args :
        question (str): The user query to be answered
    Returns: 
        answer(str): The answer to the user query
    """
    cache=cache_manager.get_cache()
    if cache is None:
        raise ValueError("Cache is empty. Please preload knowledge first.")
    
    inputs=tokenizer(question , return_tensors="pt")
    with torch.no_grad():
        outputs=model.generate(**inputs  , past_key_values=cache  , max_new_tokens=200)
    
    answer=tokenizer.decode(outputs[0] , skip_special_tokens=True)
    return answer

    
    
    
        
