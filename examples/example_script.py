import os#导入os模块，用于操作文件和目录

import translation_agent as ta#导入翻译模块，用于翻译文本


if __name__ == "__main__":#主函数
    #English to Chinese translation for China
    # source_lang, target_lang, country = "English", "Chinese", "China"
    #中文转英文，国家为美国
    source_lang, target_lang, country = "Chinese", "English", "United States"#源语言，目标语言，国家
    # relative_path = "sample-texts/sample-short1.txt"
    relative_path = "sample-texts/sample-chinese1.txt"#相对路径，用于读取文本
    script_dir = os.path.dirname(os.path.abspath(__file__))#获取当前文件的绝对路径，然后获取当前文件的目录

    full_path = os.path.join(script_dir, relative_path)#拼接路径，获取文件的绝对路径

    with open(full_path, encoding="utf-8") as file:#打开文件，编码格式为utf-8
        source_text = file.read()#读取文件内容，赋值给source_text

    print(f"Source text:\n\n{source_text}\n------------\n")#打印源文本，换行

    translation = ta.translate(#调用翻译函数
        source_lang=source_lang,#源语言
        target_lang=target_lang,#目标语言
        source_text=source_text,#源文本
        country=country,#国家
    )

    print(f"Translation:\n\n{translation}")#打印翻译结果，换行
