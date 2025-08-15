import requests
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# -----------------------------
# 1️⃣ جلب قائمة الوصفات اليابانية
# -----------------------------
url_list = "https://www.themealdb.com/api/json/v1/1/filter.php?a=Japanese"
response = requests.get(url_list)
meals = response.json()['meals']

all_recipes = []

for meal in meals:
    meal_id = meal['idMeal']
    url_detail = f"https://www.themealdb.com/api/json/v1/1/lookup.php?i={meal_id}"
    detail = requests.get(url_detail).json()['meals'][0]
    
    # تجهيز نص واحد لكل وصفة
    recipe_text = f"Name: {detail['strMeal']}\n"
    recipe_text += f"Category: {detail['strCategory']}\n"
    recipe_text += f"Area: {detail['strArea']}\n"
    recipe_text += "Ingredients:\n"
    
    for i in range(1, 21):
        ingredient = detail.get(f'strIngredient{i}')
        measure = detail.get(f'strMeasure{i}')
        if ingredient and ingredient.strip():
            recipe_text += f"- {ingredient} : {measure}\n"
    
    recipe_text += f"\nInstructions:\n{detail['strInstructions']}\n"
    
    all_recipes.append(recipe_text)

print(f"Got {len(all_recipes)} Japanese recipes!")

# -----------------------------
# 2️⃣ تهيئة ChromaDB
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="japanese_recipes")

# -----------------------------
# 3️⃣ تحميل نموذج embeddings
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# 4️⃣ تحويل كل وصفة إلى embedding وحفظها
# -----------------------------
for recipe_text in all_recipes:
    embedding = model.encode([recipe_text])
    doc_id = str(uuid.uuid4())
    collection.add(
        ids=[doc_id],
        documents=[recipe_text],
        embeddings=embedding.tolist()
    )

print(f"All {len(all_recipes)} Japanese recipes saved to ChromaDB!")
