import os#导入os模块,用于读取环境变量
from typing import List#导入List模块,用于定义列表
from typing import Union#导入Union模块,用于定义联合类型

import openai#导入openai模块,用于调用openai的接口
import tiktoken#导入tiktoken模块,用于对文本进行分词
from dotenv import load_dotenv#导入load_dotenv模块,用于读取环境变量
from icecream import ic#导入ic模块,用于调试
from langchain_text_splitters import RecursiveCharacterTextSplitter#导入RecursiveCharacterTextSplitter模块,用于对文本进行分割
from openai import OpenAI#导入OpenAI模块,用于调用openai的接口

load_dotenv()#读取环境变量
API_KEY = os.getenv("OPENAI_API_KEY")#获取openai的API_KEY
API_URL = os.getenv("OPENAI_API_URL")#获取openai的API_URL
#调用OpenAI接口
client = OpenAI(
    api_key = API_KEY,#设置openai的API_KEY
    base_url = API_URL#设置openai的API_URL
    )
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS_PER_CHUNK = (#设置文本的最大长度
    1000  # if text is more than this many tokens, we'll break it up into
)
# discrete chunks to translate one chunk at a time

def get_completion(#定义get_completion函数,用于生成文本
    prompt: str,#定义prompt参数,用于接收用户的输入
    system_message: str = "You are a helpful assistant.",#定义system_message参数,用于设置助手的上下文
    # model: str = "gpt-4-turbo",
    model: str = "gpt-3.5-turbo",#定义model参数,用于设置openai的模型
    # model: str = "claude-3-5-sonnet-20240620",
    temperature: float = 0.3,#定义temperature参数,控制生成文本的随机性
    json_mode: bool = False,#定义json_mode参数,用于设置是否返回json格式的响应
) -> Union[str, dict]:#定义返回值类型,返回字符串或字典
    """
        Generate a completion using the OpenAI API.#使用OpenAI API生成文本

    Args:#参数
        prompt (str): The user's prompt or query.#用户的提示或查询
        system_message (str, optional): The system message to set the context for the assistant.#设置助手的上下文
            Defaults to "You are a helpful assistant.".#默认为"You are a helpful assistant."
        model (str, optional): The name of the OpenAI model to use for generating the completion.#用于生成完成的OpenAI模型的名称
            Defaults to "gpt-4-turbo".#默认为"gpt-4-turbo"
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.#用于控制生成文本的随机性的采样温度
            Defaults to 0.3.#默认为0.3
        json_mode (bool, optional): Whether to return the response in JSON format.#是否以JSON格式返回响应
            Defaults to False.#默认为False

    Returns:#返回值
        Union[str, dict]: The generated completion.#生成的完成，返回字符串或字典
            If json_mode is True, returns the complete API response as a dictionary.#如果json_mode为True，则返回完整的API响应作为字典
            If json_mode is False, returns the generated text as a string.#如果json_mode为False，则返回生成的文本作为字符串
    """

    if json_mode:#如果json_mode为True，则返回完整的API响应作为字典
        response = client.chat.completions.create(#调用openai的chat.completions.create接口，生成文本
            model=model,#设置openai的模型，gpt-4-turbo
            temperature=temperature,#设置生成文本的随机性的采样温度，0.3
            top_p=1,#设置生成文本的top_p参数，1，表示返回最可能的文本
            response_format={"type": "json_object"},#设置生成文本的响应格式，json_object，返回json格式的响应
            messages=[#设置生成文本的消息，包括系统消息和用户消息
                {"role": "system", "content": system_message},#系统消息，助手的上下文
                {"role": "user", "content": prompt},#用户消息，用户的输入，提示或查询
            ],
        )
        return response.choices[0].message.content#返回生成的文本
    else:#如果json_mode为False，则返回生成的文本作为字符串
        response = client.chat.completions.create(#调用openai的chat.completions.create接口，生成文本
            model=model,#设置openai的模型，gpt-4-turbo
            temperature=temperature,#设置生成文本的随机性的采样温度，0.3
            top_p=1,#设置生成文本的top_p参数，1，表示返回最可能的文本
            messages=[#设置生成文本的消息，包括系统消息和用户消息
                {"role": "system", "content": system_message},#系统消息，助手的上下文
                {"role": "user", "content": prompt},#用户消息，用户的输入，提示或查询
            ],
        )
        return response.choices[0].message.content#返回生成的文本


def one_chunk_initial_translation(#定义one_chunk_initial_translation函数，用于将整个文本作为一个块进行翻译
    source_lang: str, target_lang: str, source_text: str#定义参数，source_lang为源语言，target_lang为目标语言，source_text为源文本
) -> str:#返回值为字符串，翻译后的文本
    """
    Translate the entire text as one chunk using an LLM.#使用LLM将整个文本作为一个块进行翻译

    Args:#参数
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language for translation.#目标语言，翻译的目标语言
        source_text (str): The text to be translated.#要翻译的文本，源文本

    Returns:
        str: The translated text.#返回值为字符串，翻译后的文本
    """
    #设置系统消息，内容是用户是专业的语言学家，专门从源语言到目标语言的翻译
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    #设置翻译提示，内容是这是一个从源语言到目标语言的翻译，请为这段文本提供目标语言的翻译。不要提供任何解释或文本，除了翻译。
    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    prompt = translation_prompt.format(source_text=source_text)#设置提示，源文本

    translation = get_completion(prompt, system_message=system_message)#调用get_completion函数，生成文本

    return translation#返回生成的文本


def one_chunk_reflect_on_translation(#定义one_chunk_reflect_on_translation函数，用于反思翻译
    source_lang: str,#定义参数，源语言
    target_lang: str,#定义参数，目标语言
    source_text: str,#定义参数，源文本
    translation_1: str,#定义参数，翻译后的文本
    country: str = "",#定义参数，国家
) -> str:#返回值为字符串
    """
    Use an LLM to reflect on the translation, treating the entire text as one chunk.#使用LLM反思翻译，将整个文本作为一个块

    Args:
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language of the translation.#目标语言，翻译的目标语言
        source_text (str): The original text in the source language.#源文本，源语言的原始文本
        translation_1 (str): The initial translation of the source text.#翻译后的文本，源文本的初始翻译
        country (str): Country specified for target language.#国家，指定目标语言

    Returns:
        str: The LLM's reflection on the translation, providing constructive criticism and suggestions for improvement.#返回值为字符串，LLM对翻译的反思，提供建设性的批评和改进建议
    """
    #设置系统消息，内容是用户是专业的语言学家，专门从源语言到目标语言的翻译
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    prompt = reflection_prompt.format(#设置提示，源文本，翻译后的文本
        source_lang=source_lang,#源语言，文本的源语言
        target_lang=target_lang,#目标语言，翻译的目标语言
        source_text=source_text,#源文本，源语言的原始文本
        translation_1=translation_1,#翻译后的文本，源文本的初始翻译
    )
    reflection = get_completion(prompt, system_message=system_message)#调用get_completion函数，生成文本
    return reflection#返回生成的文本


def one_chunk_improve_translation(#定义one_chunk_improve_translation函数，用于改进翻译
    source_lang: str,#定义参数，源语言
    target_lang: str,#定义参数，目标语言
    source_text: str,#定义参数，源文本
    translation_1: str,#定义参数，翻译后的文本
    reflection: str,#定义参数，反思
) -> str:#返回值为字符串
    """
    Use the reflection to improve the translation, treating the entire text as one chunk.#使用反思来改进翻译，将整个文本作为一个块

    Args:
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language for the translation.#目标语言，翻译的目标语言
        source_text (str): The original text in the source language.#源文本，源语言的原始文本
        translation_1 (str): The initial translation of the source text.#翻译后的文本，源文本的初始翻译
        reflection (str): Expert suggestions and constructive criticism for improving the translation.#反思，用于改进翻译的专家建议和建设性批评

    Returns:
        str: The improved translation based on the expert suggestions.#返回值为字符串，基于专家建议的改进翻译
    """
    #设置系统消息，内容是用户是专业的语言学家，专门从源语言到目标语言的翻译
    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    #设置提示，内容是您的任务是仔细阅读，然后编辑从源语言到目标语言的翻译，考虑专家建议和建设性的批评
    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""
    
    translation_2 = get_completion(prompt, system_message)#调用get_completion函数，生成文本

    return translation_2#返回生成的文本


def one_chunk_translate_text(#定义one_chunk_translate_text函数，用于翻译文本
    source_lang: str, target_lang: str, source_text: str, country: str = ""#定义参数，源语言，目标语言，源文本，国家
) -> str:#返回值为字符串
    """
    Translate a single chunk of text from the source language to the target language.#将源语言的单个文本块翻译为目标语言

    This function performs a two-step translation process:#此函数执行两步翻译过程
    1. Get an initial translation of the source text.#获取源文本的初始翻译
    2. Reflect on the initial translation and generate an improved translation.#反思初始翻译并生成改进的翻译

    Args:#参数
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language for the translation.#目标语言，翻译的目标语言
        source_text (str): The text to be translated.#要翻译的文本
        country (str): Country specified for target language.#国家，指定目标语言
    Returns:
        str: The improved translation of the source text.#返回值为字符串，源文本的改进翻译
    """
    translation_1 = one_chunk_initial_translation(#调用one_chunk_initial_translation函数，将整个文本作为一个块进行翻译
        source_lang, target_lang, source_text#源语言，目标语言，源文本
    )

    reflection = one_chunk_reflect_on_translation(#调用one_chunk_reflect_on_translation函数，用于反思翻译
        source_lang, target_lang, source_text, translation_1, country#源语言，目标语言，源文本，翻译后的文本，国家
    )
    translation_2 = one_chunk_improve_translation(#调用one_chunk_improve_translation函数，用于改进翻译
        source_lang, target_lang, source_text, translation_1, reflection#源语言，目标语言，源文本，翻译后的文本，反思
    )

    return translation_2#返回生成的文本S


def num_tokens_in_string(#定义num_tokens_in_string函数，用于计算字符串中的标记数
    input_str: str, encoding_name: str = "cl100k_base"#定义参数，输入字符串，编码名称
) -> int:
    """
    Calculate the number of tokens in a given string using a specified encoding.#使用指定的编码计算给定字符串中的标记数

    Args:
        str (str): The input string to be tokenized.#要标记化的输入字符串
        encoding_name (str, optional): The name of the encoding to use. Defaults to "cl100k_base",#要使用的编码的名称，默认为"cl100k_base"
            which is the most commonly used encoder (used by GPT-4).#这是最常用的编码器（GPT-4使用）

    Returns:
        int: The number of tokens in the input string.#输入字符串中的标记数

    Example:
        >>> text = "Hello, how are you?"#定义文本,内容是"Hello, how are you?"
        >>> num_tokens = num_tokens_in_string(text)#调用num_tokens_in_string函数，计算字符串中的标记数
        >>> print(num_tokens)#打印标记数
        5#输出结果为5
    """
    encoding = tiktoken.get_encoding(encoding_name)#调用get_encoding函数，获取编码
    num_tokens = len(encoding.encode(input_str))#计算字符串中的标记数
    return num_tokens#返回标记数


def multichunk_initial_translation(#定义multichunk_initial_translation函数，用于将文本分成多个块进行翻译
    source_lang: str, target_lang: str, source_text_chunks: List[str]#定义参数，源语言，目标语言，源文本块
) -> List[str]:#返回值为列表
    """
    Translate a text in multiple chunks from the source language to the target language.#将文本分成多个块进行翻译，从源语言到目标语言

    Args:
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language for translation.#目标语言，翻译的目标语言
        source_text_chunks (List[str]): A list of text chunks to be translated.#要翻译的文本块的列表

    Returns:
        List[str]: A list of translated text chunks.#返回值为列表，翻译后的文本块
    """
    #设置系统消息，内容是用户是专业的语言学家，专门从源语言到目标语言的翻译
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    #设置翻译提示，内容是这是一个从源语言到目标语言的翻译，请为这段文本提供目标语言的翻译。不要提供任何解释或文本，除了翻译。
    translation_prompt = """Your task is provide a professional translation from {source_lang} to {target_lang} of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.
"""

    translation_chunks = []#定义空列表，用于存储翻译后的文本块
    for i in range(len(source_text_chunks)):#遍历源文本块,将文本分成多个块进行翻译
        # Will translate chunk i
        tagged_text = (#设置标记文本，内容是将源文本块分成多个块进行翻译
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        #设置提示，源语言，目标语言，标记文本，要翻译的文本块
        prompt = translation_prompt.format(#设置提示，源文本块
            source_lang=source_lang,#源语言，文本的源语言
            target_lang=target_lang,#目标语言，翻译的目标语言
            tagged_text=tagged_text,#标记文本，将源文本块分成多个块进行翻译
            chunk_to_translate=source_text_chunks[i],#要翻译的文本块
        )
        
        translation = get_completion(prompt, system_message=system_message)#调用get_completion函数，生成文本
        translation_chunks.append(translation)#将生成的文本添加到列表中

    return translation_chunks#返回生成的文本列表


def multichunk_reflect_on_translation(#定义multichunk_reflect_on_translation函数，用于反思翻译
    source_lang: str,#定义参数，源语言
    target_lang: str,#定义参数，目标语言
    source_text_chunks: List[str],#定义参数，源文本块
    translation_1_chunks: List[str],#定义参数，翻译后的文本块
    country: str = "",#定义参数，国家
) -> List[str]:#返回值为列表
    """
    Provides constructive criticism and suggestions for improving a partial translation.#提供建设性的批评和建议，以改进部分翻译

    Args:
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language of the translation.#目标语言，翻译的目标语言
        source_text_chunks (List[str]): The source text divided into chunks.#源文本块，将源文本分成多个块
        translation_1_chunks (List[str]): The translated chunks corresponding to the source text chunks.#翻译后的文本块，与源文本块对应
        country (str): Country specified for target language.#国家，指定目标语言

    Returns:
        List[str]: A list of reflections containing suggestions for improving each translated chunk.#返回值为列表，包含每个翻译文本块的改进建议
    """
    #设置系统消息，内容是用户是专业的语言学家，专门从源语言到目标语言的翻译
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
    #设置反思提示，内容是您的任务是仔细阅读，然后编辑从源语言到目标语言的翻译，考虑专家建议和建设性的批评
    if country != "":
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:#如果没有指定国家
        #设置反思提示，内容是您的任务是仔细阅读，然后编辑从源语言到目标语言的翻译，考虑专家建议和建设性的批评
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection_chunks = []#定义空列表，用于存储反思
    for i in range(len(source_text_chunks)):#遍历源文本块，将文本分成多个块进行翻译
        # Will translate chunk i
        tagged_text = (#设置标记文本，内容是将源文本块分成多个块进行翻译
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        if country != "":#如果指定了国家,则设置反思提示，源文本块，翻译后的文本块，国家
            prompt = reflection_prompt.format(#设置反思提示，源文本块
                source_lang=source_lang,#源语言，文本的源语言
                target_lang=target_lang,#目标语言，翻译的目标语言
                tagged_text=tagged_text,#标记文本，将源文本块分成多个块进行翻译
                chunk_to_translate=source_text_chunks[i],#要翻译的文本块
                translation_1_chunk=translation_1_chunks[i],#翻译后的文本块
                country=country,#国家
            )
        else:#如果没有指定国家,则设置反思提示，源文本块，翻译后的文本块
            prompt = reflection_prompt.format(#设置反思提示，源文本块
                source_lang=source_lang,#源语言，文本的源语言
                target_lang=target_lang,#目标语言，翻译的目标语言
                tagged_text=tagged_text,#标记文本，将源文本块分成多个块进行翻译
                chunk_to_translate=source_text_chunks[i],#要翻译的文本块
                translation_1_chunk=translation_1_chunks[i],#翻译后的文本块
            )
        #调用get_completion函数，生成文本
        reflection = get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)#将生成的文本添加到列表中

    return reflection_chunks#返回生成的文本列表


def multichunk_improve_translation(#定义multichunk_improve_translation函数，用于改进翻译
    source_lang: str,#定义参数，源语言
    target_lang: str,#定义参数，目标语言
    source_text_chunks: List[str],#定义参数，源文本块
    translation_1_chunks: List[str],#定义参数，翻译后的文本块
    reflection_chunks: List[str],#定义参数，反思
) -> List[str]:#返回值为列表
    """
    Improves the translation of a text from source language to target language by considering expert suggestions.#通过考虑专家建议，改进从源语言到目标语言的文本的翻译

    Args:
        source_lang (str): The source language of the text.#源语言，文本的源语言
        target_lang (str): The target language for translation.#目标语言，翻译的目标语言
        source_text_chunks (List[str]): The source text divided into chunks.#源文本块，将源文本分成多个块
        translation_1_chunks (List[str]): The initial translation of each chunk.#每个文本块的初始翻译
        reflection_chunks (List[str]): Expert suggestions for improving each translated chunk.#用于改进每个翻译文本块的专家建议

    Returns:
        List[str]: The improved translation of each chunk.#返回值为列表，每个文本块的改进翻译
    """
    #设置系统消息，内容是用户是专业的语言学家，专门从源语言到目标语言的翻译
    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    #设置改进提示，内容是您的任务是仔细阅读，然后编辑从源语言到目标语言的翻译，考虑专家建议和建设性的批评
    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
account a set of expert suggestions and constructive criticisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, is as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation of the indicated part and nothing else."""

    translation_2_chunks = []#定义空列表，用于存储改进翻译后的文本块
    for i in range(len(source_text_chunks)):#遍历源文本块，将文本分成多个块进行翻译
        # Will translate chunk i
        tagged_text = (#设置标记文本，内容是将源文本块分成多个块进行翻译
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = improvement_prompt.format(#设置改进提示，源文本块
            source_lang=source_lang,#源语言，文本的源语言
            target_lang=target_lang,#目标语言，翻译的目标语言
            tagged_text=tagged_text,#标记文本，将源文本块分成多个块进行翻译
            chunk_to_translate=source_text_chunks[i],#要翻译的文本块
            translation_1_chunk=translation_1_chunks[i],#翻译后的文本块
            reflection_chunk=reflection_chunks[i],#反思
        )
        #调用get_completion函数，生成文本
        translation_2 = get_completion(prompt, system_message=system_message)
        translation_2_chunks.append(translation_2)#将生成的文本添加到列表中

    return translation_2_chunks#返回生成的文本列表


def multichunk_translation(#定义multichunk_translation函数，用于将文本分成多个块进行翻译
    source_lang, target_lang, source_text_chunks, country: str = ""#定义参数，源语言，目标语言，源文本块，国家
):
    """
    Improves the translation of multiple text chunks based on the initial translation and reflection.#根据初始翻译和反思，改进多个文本块的翻译

    Args:
        source_lang (str): The source language of the text chunks.#源语言，文本块的源语言
        target_lang (str): The target language for translation.#目标语言，翻译的目标语言
        source_text_chunks (List[str]): The list of source text chunks to be translated.#要翻译的源文本块的列表
        translation_1_chunks (List[str]): The list of initial translations for each source text chunk.#每个源文本块的初始翻译的列表
        reflection_chunks (List[str]): The list of reflections on the initial translations.#初始翻译的反思列表
        country (str): Country specified for target language#国家，指定目标语言
    Returns:
        List[str]: The list of improved translations for each source text chunk.#返回值为列表，每个源文本块的改进翻译
    """

    translation_1_chunks = multichunk_initial_translation(#调用multichunk_initial_translation函数，将文本分成多个块进行翻译
        source_lang, target_lang, source_text_chunks#源语言，目标语言，源文本块
    )

    reflection_chunks = multichunk_reflect_on_translation(#调用multichunk_reflect_on_translation函数，用于反思翻译
        source_lang,#源语言
        target_lang,#目标语言
        source_text_chunks,#源文本块
        translation_1_chunks,#翻译后的文本块
        country,#国家
    )

    translation_2_chunks = multichunk_improve_translation(#调用multichunk_improve_translation函数，用于改进翻译
        source_lang,#源语言
        target_lang,#目标语言
        source_text_chunks,#源文本块
        translation_1_chunks,#翻译后的文本块
        reflection_chunks,#反思
    )

    return translation_2_chunks#返回生成的文本列表


def calculate_chunk_size(token_count: int, token_limit: int) -> int:#定义calculate_chunk_size函数，用于计算块大小
    """
    Calculate the chunk size based on the token count and token limit.#根据标记计数和标记限制计算块大小

    Args:
        token_count (int): The total number of tokens.#标记计数，标记的总数
        token_limit (int): The maximum number of tokens allowed per chunk.#标记限制，每个块允许的最大标记数

    Returns:
        int: The calculated chunk size.#返回值为整数，计算得到的块大小

    Description:
        This function calculates the chunk size based on the given token count and token limit.#此函数根据给定的标记计数和标记限制计算块大小
        If the token count is less than or equal to the token limit, the function returns the token count as the chunk size.#如果标记计数小于或等于标记限制，则函数将标记计数作为块大小返回
        Otherwise, it calculates the number of chunks needed to accommodate all the tokens within the token limit.#否则，它计算需要的块数以容纳标记限制内的所有标记
        The chunk size is determined by dividing the token limit by the number of chunks.#块大小通过将标记限制除以块数来确定
        If there are remaining tokens after dividing the token count by the token limit,#如果将标记计数除以标记限制后还有剩余标记
        the chunk size is adjusted by adding the remaining tokens divided by the number of chunks.#通过将剩余标记除以块数来调整块大小

    Example:
        >>> calculate_chunk_size(1000, 500)#调用calculate_chunk_size函数，计算块大小
        500
        >>> calculate_chunk_size(1530, 500)#调用calculate_chunk_size函数，计算块大小
        389
        >>> calculate_chunk_size(2242, 500)#调用calculate_chunk_size函数，计算块大小
        496
    """

    if token_count <= token_limit:#如果标记计数小于或等于标记限制
        return token_count#返回标记计数

    num_chunks = (token_count + token_limit - 1) // token_limit#计算需要的块数以容纳标记限制内的所有标记
    chunk_size = token_count // num_chunks#计算块大小

    remaining_tokens = token_count % token_limit#计算剩余标记
    if remaining_tokens > 0:#如果有剩余标记
        chunk_size += remaining_tokens // num_chunks#调整块大小

    return chunk_size#返回块大小


def translate(#定义translate函数，用于翻译文本
    source_lang,#定义参数，源语言
    target_lang,#定义参数，目标语言
    source_text,#定义参数，源文本
    country,#定义参数，国家
    max_tokens=MAX_TOKENS_PER_CHUNK,#定义参数，最大标记数
):
    """Translate the source_text from source_lang to target_lang."""

    num_tokens_in_text = num_tokens_in_string(source_text)#调用num_tokens_in_string函数，计算字符串中的标记数

    ic(num_tokens_in_text)#打印标记数

    if num_tokens_in_text < max_tokens:#如果标记数小于最大标记数
        ic("Translating text as single chunk")#打印信息，将文本作为单个块进行翻译

        final_translation = one_chunk_translate_text(#调用one_chunk_translate_text函数，将整个文本作为一个块进行翻译
            source_lang, target_lang, source_text, country#源语言，目标语言，源文本，国家
        )

        return final_translation#返回生成的文本

    else:#如果标记数大于等于最大标记数
        ic("Translating text as multiple chunks")#打印信息，将文本分成多个块进行翻译

        token_size = calculate_chunk_size(#调用calculate_chunk_size函数，计算块大小
            token_count=num_tokens_in_text, token_limit=max_tokens#标记计数，标记限制
        )

        ic(token_size)#打印块大小

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(#使用tiktoken编码器创建RecursiveCharacterTextSplitter
            model_name="gpt-4",#模型名称
            chunk_size=token_size,#块大小
            chunk_overlap=0,#块重叠
        )

        source_text_chunks = text_splitter.split_text(source_text)#将文本分成多个块

        translation_2_chunks = multichunk_translation(#调用multichunk_translation函数，将文本分成多个块进行翻译
            source_lang, target_lang, source_text_chunks, country#源语言，目标语言，源文本块，国家
        )

        return "".join(translation_2_chunks)#返回生成的文本
