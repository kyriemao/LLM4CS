from IPython import embed
import re
import json


def check_length(prompt, max_length):
    n = len(prompt.split(' '))
    if n >= max_length:
        return False
    return True


class RewritePromptor:
    def __init__(self, demo_file=None, enable_cot=False) -> None:    
        
        self.instruction = "For an information-seeking dialog, please help reformulate the question into rewrite that can fully express the user's information needs without the need of context."
        self.enable_cot = enable_cot
        self.demo = self.get_demo(demo_file)
        if self.demo != "":
            self.instruction += " I will give you several example multi-turn dialogs, where each turn contains a question, a response, and a rewrite that you need to generate."
            if enable_cot:
                self.instruction += " The rewrite part begins with a sentence explaining the reason for the generated rewrite."
        if enable_cot:
            self.tail_instruction = "Now, you should give me the rewrite of the **Current Question** under the **Context**. The output format should always be: \"Rewrite: $Reason. So the question should be rewritten as: $Rewrite.\" Note that you should always try to rewrite it. Never ask for clarification or say you don't understand it in the generated rewrite. Go ahead!"
        else:
            self.tail_instruction = "Now, you should give me the rewrite of the **Current Question** under the **Context**. The output format should always be: Rewrite: $Rewrite. Note that you should always try to rewrite it. Never ask for clarification or say you don't understand it in the generated rewrite. Go ahead!"
        self.stop_tokens = ['\n']
    
    def get_demo(self, demo_file):
        try:
            with open(demo_file, "r") as f:
                demos = json.load(f)
        except:
            print("warning: No demonstration file.")
            return ""
        
        examples = []
        for demo in demos:
            turns = demo['turns']
            
            dialog = []
            for turn in turns:
                if self.enable_cot:
                    rewrite = turn['cot'] + " So the question should be rewritten as: {}".format(turn['manual_rewrite'])
                else:
                    rewrite = turn['manual_rewrite']
                turn_text = "Question: {}\nRewrite: {}\nResponse: {}".format(turn['question'], rewrite, turn['response'])         
                dialog.append(turn_text)
            dialog = "\n\n".join(dialog)
            
            examples.append(dialog)
        
        for i in range(len(examples)):
            examples[i] = "Example #{}:\n".format(i+1) + examples[i]
        
        return "\n\n".join(examples)
              
    def build_turn_prompt(self, context, current_turn):
        # context
        this_dialog = []
        if not context:
            this_dialog.append("N/A")
        else:
            for turn in context:
                this_dialog.append("Question: {}\nResponse: {}".format(turn['question'], turn['response']))
        
        this_dialog[0] = "Context:\n" + this_dialog[0]
            
        # current turn
        this_dialog.append("Current Question: " + current_turn['question'])
        this_dialog = "\n\n".join(this_dialog)  
        this_dialog = "YOUR TASK (only questions and responses may be given):\n" + this_dialog
        this_prompt = [self.instruction, self.demo, this_dialog, self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    
    def parse_returned_text(self, text):
        text = text.strip()
        if text[:9] != "Rewrite: ":
            return None
        if not self.enable_cot:
            return text[9:]
        else:
            fixed_sentence = "So the question should be rewritten as: "
            index = text.find(fixed_sentence)
            if index != -1:
                cot = text[:index]
                rewrite = text[index + len(fixed_sentence):]
                return [cot.strip(), rewrite.strip()]
            else:
                return None

class RewriteAndResponsePromptor:
    def __init__(self, demo_file, enable_cot=False) -> None:    
        
        self.instruction = "For an information-seeking dialog, please help reformulate the question into rewrite that can fully express the user's information needs without the need of context, but also generate an informative response to answer the question."
        self.enable_cot = enable_cot
        self.demo = self.get_demo(demo_file)
        if self.demo != "":
            self.instruction += " I will give you several example multi-turn dialogs, where each turn contains a question as well as a rewrite and a response that you need to generate."
            if enable_cot:
                self.instruction += " The rewrite part begins with a sentence explaining the reason for the generated rewrite."
        if enable_cot:
            self.tail_instruction = "Now, you should give me the rewrite and response of the **Current Question** under the **Context**. The output format should always be: \"Rewrite: $Reason. So the question should be rewritten as: $Rewrite\nResponse: $Response.\" Note that you should always try to rewrite it and generate an informative response. Never ask for clarification or say you don't understand it in the generated rewrite and response. Go ahead!"
        else:
            self.tail_instruction = "Now, you should give me the rewrite and response of the **Current Question** under the **Context**. The output format should always be:\nRewrite: $Rewrite\nResponse: $Response.\nNote that you should always try to rewrite it and generate an informative response. Never ask for clarification or say you don't understand it in the generated rewrite and response. Go ahead!"
        self.stop_tokens = None
                            
    def get_demo(self, demo_file):
        try:
            with open(demo_file, "r") as f:
                demos = json.load(f)
        except:
            print("warning: No demonstration file.")
            return ""
        
        examples = []
        for demo in demos:
            turns = demo['turns']
            
            dialog = []
            for turn in turns:
                if self.enable_cot:
                    rewrite = turn['cot'] + " So the question should be rewritten as: {}".format(turn['manual_rewrite'])
                else:
                    rewrite = turn['manual_rewrite']
                turn_text = "Question: {}\nRewrite: {}\nResponse: {}".format(turn['question'], rewrite, turn['response'])         
                dialog.append(turn_text)
            dialog = "\n\n".join(dialog)
            
            examples.append(dialog)
        
        for i in range(len(examples)):
            examples[i] = "Example #{}:\n".format(i+1) + examples[i]
        
        return "\n\n".join(examples)
    
    def build_turn_prompt(self, context, current_turn):
        # context
        this_dialog = []
        if not context:
            this_dialog.append("N/A")
        else:
            for turn in context:
                this_dialog.append("Question: {}\nResponse: {}".format(turn['question'], turn['response']))
        
        this_dialog[0] = "Context:\n" + this_dialog[0]
            
        # current turn
        this_dialog.append("Current Question: " + current_turn['question'])
        this_dialog = "\n\n".join(this_dialog)  
        this_dialog = "YOUR TASK (only questions and responses may be given):\n" + this_dialog
        this_prompt = [self.instruction, self.demo, this_dialog, self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    
    def parse_returned_text(self, text):
        text = text.strip()
        try:
            splits = text.split('\n')
            if splits[0][:9] != "Rewrite: " or splits[1][:10] != "Response: ":
                return None
            if self.enable_cot:
                rewrite_text = splits[0][9:]
                fixed_sentence = "So the question should be rewritten as: "
                index = rewrite_text.find(fixed_sentence)
                if index != -1:
                    cot = rewrite_text[:index]
                    rewrite = rewrite_text[index + len(fixed_sentence):]
                else:
                    return None            
                
                response = "\n".join(splits[1:])[10:]
                return [cot, rewrite, response]
            else:
                rewrite = splits[0][9:]
                response = "\n".join(splits[1:])[10:]
                return [rewrite, response]
        except:
            return None



class RewriteThenResponsePrompter:
    def __init__(self, demo_file, enable_cot=False) -> None:    
        
        self.instruction = "For an information-seeking dialog, please help generate an informative response to answer the question."
        self.enable_cot = enable_cot
        self.demo = self.get_demo(demo_file)
        if self.demo != "":
            self.instruction += " I will give you several example multi-turn dialogs, where each turn contains a question as well as a rewrite and a response that you need to generate."
            if enable_cot:
                self.instruction += " The rewrite part begins with a sentence explaining the reason for the generated rewrite."
 
        self.tail_instruction = "Now, you should give me the response of the **Current Question** under the **Context**. The output format should always be: \"Response: $Response.\" Note that you should always try to generate an informative response. Never ask for clarification or say you don't understand it in the response. Go ahead!"
        self.stop_tokens = None
                            
    def build_turn_prompt(self, context, current_turn):
        # context
        all_prompts = [self.instruction]
        if not context:
            all_prompts.append("N/A")
        else:
            for turn in context:
                all_prompts.append("Q: {}\nA: {}".format(turn['question'], turn['response']))
        
        all_prompts[1] = "Context:\n" + all_prompts[1]
            
        # current turn
        all_prompts.append("Current Question: " + current_turn['question'])
        
        this_prompt = "\n\n".join(all_prompts)
        
        while not check_length(this_prompt, 3700):
            t = 1
            all_prompts[t] = "Q: {}\nA: {}".format(context[t-1]['question'], "N/A")
            this_prompt = "\n\n".join(all_prompts)
            t += 1        
        
        this_prompt += "\n\n**Note that you should always try to generate a plausible answer. Never ask for clarification or say you don't understand it in the generated response.** Now, go ahead!"
        
        return this_prompt
    
    def parse_returned_text(self, text):
        text = text.strip()
        if text[:10] != "Response: ":
            return None
        return text[10:]
    



class OneShotRewriteThenResponsePrompter:
    def __init__(self) -> None:    
        
        self.instruction = "For an information-seeking dialog. Please help generate an appropriate response to answer the question.\n\nI will give you an example multi-turn dialog, where each turn contains a question and the corresponding response.\n\n# An Example Multi-turn Dialog:\n\nQuestion: What should I consider when buying a phone?\nResponse: The design of the phone and the overall look and feel of the phone are very important. You should be comfortable with the way the phone looks and feels when you hold it in your hand. In addition, don't be afraid to get a phone from a different manufacturer than you're used to. Consider an older, used, or refurbished phone to save money. Sometimes a year-old or even a two-year-old device might give you everything you need.\n\nQuestion: Cool. Which one would you recommend?\nResponse: If you want something more affordable, cheaper, but has the latest in Android and that comes with 5000mAh of battery, then the Motorola Moto G7 Power is an excellent phone to consider. If you want an Android smartphone that's cheaper, more affordable, and you still find the Moto G7 Power a bit off your budget, then I highly recommend the Samsung Galaxy A10e.\n\nQuestion: Tell me more about the first one.\nResponse: It sports an 8-megapixel camera on the front for selfies with an f/2.2 aperture and a pixel size of 1.12-micron. Motorola Moto G7 Power is based on Android 9.0 and packs 32GB of inbuilt storage that can be expanded via a microSD card (up to 512GB). The Motorola Moto G7 Power is a dual-SIM (GSM and GSM) smartphone that accepts Nano-SIM and Nano-SIM cards. The Motorola Moto G7 Power measures 160.83 x 76.00 x 9.40mm (height x width x thickness) and weighs 198.00 grams.\n\nQuestion: How much cheaper is the A10e?\nResponse: The current lowest price found for Samsung Galaxy A10e is 7,499 and for Motorola Moto G7 is 13990. The details of both of these products were last updated on Jun 02, 2021.\n\nQuestion: Wow, that's almost half the cost! Can you compare them?\nResponse: For the past few years, Motorola's G-series has consistently ranked as a favourite budget phone among users and the G7 continues this legacy as an exceptional phone for its price. All in all, people still love the Moto G7. It's a fantastic, affordable phone, and it won awards for its inexpensive price, good camera and reliable performance. But the Galaxy A50 is better, even if it's a bit more expensive. \n\n# Now, you should give me the response of the **Current Question** for the following dialog. The output format should be: Response: $Response."
        self.stop_tokens = None
                            
    def build_turn_prompt(self, context, current_turn):
        # context
        all_prompts = [self.instruction]
        if not context:
            all_prompts.append("N/A")
        else:
            for turn in context:
                all_prompts.append("Q: {}\nA: {}".format(turn['question'], turn['response']))
        
        all_prompts[1] = "Context:\n" + all_prompts[1]
            
        # current turn
        all_prompts.append("Current Question: " + current_turn['question'])
        
        this_prompt = "\n\n".join(all_prompts)
        
        while not check_length(this_prompt, 3700):
            t = 1
            all_prompts[t] = "Q: {}\nA: {}".format(context[t-1]['question'], "N/A")
            this_prompt = "\n\n".join(all_prompts)
            t += 1        
        
        this_prompt += "\n\n**Note that you should always try to generate a plausible answer. Never ask for clarification or say you don't understand it in the generated response.** Now, go ahead!"
        
        return this_prompt
    
    def parse_returned_text(self, text):
        text = text.strip()
        if text[:10] != "Response: ":
            return None
        return text[10:]
    
