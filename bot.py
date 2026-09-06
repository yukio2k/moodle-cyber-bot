import os
import zipfile
import logging
import asyncio
from aiogram import Bot, Dispatcher, F, types
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Получаем токены из переменных окружения (безопасно для GitHub и серверов)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

if not TELEGRAM_TOKEN or not GOOGLE_KEY:
    logging.error("Не найдены переменные окружения TELEGRAM_TOKEN или GOOGLE_API_KEY!")

# Пути к файлам и папкам
ZIP_PATH = "moodle_docs.zip"
DOCS_DIR = "moodle_docs"
DB_DIR = "./chroma_db"

# 1. Автоматическая распаковка архива с лекциями при первом запуске
if os.path.exists(ZIP_PATH) and not os.path.exists(DOCS_DIR):
    logging.info("Распаковываем архив с лекциями...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DOCS_DIR)
    logging.info("Архив успешно распакован!")

# 2. Инициализация эмбеддингов
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_KEY
)

# 3. Автоматическая индексация лекций или загрузка существующей базы
if not os.path.exists(DB_DIR) and os.path.exists(DOCS_DIR):
    logging.info("Создаем векторную базу данных из лекций...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    logging.info("База данных успешно создана и сохранена!")
else:
    try:
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        logging.info("Векторная база данных успешно загружена.")
    except Exception as e:
        logging.warning(f"Не удалось загрузить базу данных: {e}")
        vectorstore = None

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None

# Инициализация языковой модели Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    google_api_key=GOOGLE_KEY
)

system_prompt = (
    "Ты — личный ИИ-ассистент и преподаватель-наставник по кибербезопасности для студента. "
    "Твоя задача — помочь ему разобраться в материале, объяснить сложные концепции простыми словами и помогать с заданиями.\n\n"
    "У тебя есть доступ к материалам лекций студента (контекст ниже). "
    "Используй их в первую очередь. Если информации недостаточно, смело привлекай свои знания и интернет, "
    "чтобы дать исчерпывающий ответ.\n\n"
    "Материалы из лекций:\n{context}\n\n"
    "Вопрос студента: {input}"
)

# Инициализация Телеграм бота
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(F.text == "/start")
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я твой облачный ИИ-помощник по кибербезопасности 🛡\n"
        "Я готов отвечать на вопросы по твоим лекциям и учебе 24/7!"
    )

@dp.message(F.text)
async def handle_message(message: types.Message):
    await message.bot.send_chat_action(message.chat.id, "typing")
    try:
        user_query = message.text
        context_text = "В локальных лекциях прямого упоминания нет."
        
        if retriever:
            try:
                docs = retriever.invoke(user_query)
                if docs:
                    context_text = "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logging.error(f"Ошибка поиска в базе: {e}")
        
        formatted_prompt = system_prompt.format(context=context_text, input=user_query)
        response = llm.invoke(formatted_prompt)
        await message.answer(response.content)
        
    except Exception as e:
        logging.error(f"Ошибка обработки запроса: {e}")
        await message.answer("Произошла ошибка при обработке запроса.")

async def main():
    logging.info("Запуск Telegram бота...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())