from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

text = """
Albert Camus's The Myth of Sisyphus is a philosophical essay published in 1942 that explores the meaning of human existence. Camus introduces the concept of the absurd, which arises from the conflict between human beings' desire to find meaning and the universe's silence. People naturally seek answers to questions such as why we exist, what our purpose is, and why suffering occurs. However, the universe does not provide clear or final answers to these questions. Camus argues that we must recognize this uncertainty rather than depend on the belief that life has a predetermined meaning.
Camus also examines how people respond to the absurd. He believes that the absence of a universal purpose does not mean that life is worthless. Instead, we should accept uncertainty and continue living with awareness, freedom, and passion. He encourages people to create their own values, make meaningful choices, and appreciate their experiences without waiting for absolute answers. For Camus, revolt means continually facing the absurd without surrendering to despair or escaping into false certainty. This attitude allows individuals to live consciously and take responsibility for their own lives.
The central symbol of the book is Sisyphus, a figure from Greek mythology who is condemned to push a heavy rock up a mountain forever, only to see it roll back down each time he approaches the summit. His endless task represents the repetitive struggles of human life. Although his work appears meaningless, Camus imagines Sisyphus as capable of understanding and accepting his fate. His famous conclusion, “One must imagine Sisyphus happy,” expresses the idea that people can find freedom and dignity by facing their circumstances honestly. In simple words, Camus teaches us that even if life has no universal meaning, we can still live fully, appreciate our experiences, and find our own reasons to continue.
"""
# Fixed size chunking
fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
)
textchunk = fixed.split_text(text)
for i, chunk in enumerate(textchunk, 1):
    pass
    # print(f"\nChunk {i}")
    # print(chunk)
    # print("------------------------")


# Paragraph chunking
paragraph = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0,
)
paratext = paragraph.split_text(text)
for i, chunk in enumerate(paratext, 1):
    pass
    # print(f"\n Chunk {i}")
    # print(chunk)
    # print("------------------------")


# Recursive chunking
recursive = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)
recursivetext = recursive.split_text(text)
for i, chunk in enumerate(recursivetext,1):
    # pass
    print(f"\n Recursive chunk {i}")
    print(chunk)
    print("------------------------")
