from dataclasses import dataclass

DEFAULT_FALLBACK = "Sorry, I can’t help with that request. I can help with a safe alternative or a benign version."

@dataclass
class FallbackDecision:
    action:str
    reason:str
    retry_count:int
    max_retries:int
    fallback_text:str
    exhausted:bool

class FallbackRouter:
    def __init__(self,max_retries:int=1,strategy:str='canned_refusal',safe_override_text:str|None=None):
        self.max_retries=max_retries
        self.strategy=strategy
        self.safe_override_text=safe_override_text or DEFAULT_FALLBACK

    def route(self,*,retry_count:int,reason:str,unchanged_retry_attempted:bool=False)->FallbackDecision:
        exhausted = retry_count >= self.max_retries or unchanged_retry_attempted
        if not exhausted:
            return FallbackDecision('retry','retry_budget_available',retry_count,self.max_retries,'',False)
        if self.strategy=='abort':
            return FallbackDecision('abort',reason,retry_count,self.max_retries,'',True)
        if self.strategy=='safe_override':
            return FallbackDecision('safe_override',reason,retry_count,self.max_retries,self.safe_override_text,True)
        return FallbackDecision('canned_refusal',reason,retry_count,self.max_retries,DEFAULT_FALLBACK,True)
