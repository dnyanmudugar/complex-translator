from complex_translator import translate

src_lang = "en"
tgt_lang = "es"

# Initialize the translator (defaults to English 'en' to Spanish 'es')
translation = translate(text=text, src_lang, tgt_lang)

# Translate text
text = "Hello, welcome to my custom offline translator!"

print(translation)
# Output: Hola, bienvenido a mi traductor sin conexión personalizado!