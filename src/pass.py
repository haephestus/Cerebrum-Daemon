import bcrypt

password = "kingbodgan"
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

print(password_hash)
