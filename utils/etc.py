import time


def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def stream_data(content):
    for word in content.split(" "):
        yield word + " "
        time.sleep(0.025)