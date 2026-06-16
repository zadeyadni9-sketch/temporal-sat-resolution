import os

root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root, "src")
print("root:", root)
print("src_dir:", src_dir)
print("src exists:", os.path.isdir(src_dir))
if os.path.isdir(src_dir):
    print("src contents:", os.listdir(src_dir))
else:
    print("NO src directory found")