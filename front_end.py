import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from customtkinter import *
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image
import sqlite3
from PIL import Image as PILImage
import webbrowser

class AttentionGate(nn.Module):
    def __init__(self, channels):
        super(AttentionGate, self).__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer1 = nn.Linear(channels, channels)
        self.linear_layer2 = nn.Linear(channels, channels)

    def forward(self, x):
        a, b, c, d = x.shape
        pooled = self.globalpool(x).view(a, b)
        weights = F.relu(self.linear_layer1(pooled))
        weights = torch.sigmoid(self.linear_layer2(weights))
        weights = weights.view(a, b, 1, 1)
        return x * weights

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.attn = AttentionGate(256)
        self.fc1 = nn.Linear(256 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.attn(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = model()
net.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
net.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

conn = sqlite3.connect("users.db")
cursor = conn.cursor()
root = CTk()
root.geometry("600x600")

start = 0
end = 10
username_enter = None
password_enter = None

def call_model():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")], title="Select an image")
    if file_path:
        image = PILImage.open(file_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        output = net(image)
        _, predicted = torch.max(output, 1)
        if predicted.item() == 1:
            cursor.execute("UPDATE users SET score = score + 10 WHERE username = ?", (username_enter.get(),))
            conn.commit()
            messagebox.showinfo("Result", "Trash detected. Points added!")
        else:
            messagebox.showinfo("Result", "No trash detected.")

def open_recycling_finder():
    webbrowser.open("https://search.earth911.com/")

def change():
    global start, end
    start += 10
    end += 10

def previous_page():
    global start, end
    if start >= 5:
        start -= 5
        end -= 5

def home():
    window0 = CTkFrame(root, width=600, height=600, fg_color="#ccc5b9")
    window0.place(x=0, y=0)
    leaderboard = CTkFrame(root, fg_color="#fffcf2", width=300, height=400, border_color="#252422", border_width=3)
    leaderboard.place(x=150, y=100)
    label = CTkLabel(leaderboard, text="Leaderboard:", fg_color="transparent")
    label.place(relx=0.5, y=10, anchor="n")
    cursor.execute("SELECT username, score FROM users ORDER BY score DESC")
    rows = cursor.fetchall()
    visible = rows[start:end]
    for i in range(len(visible)):
        CTkLabel(master=leaderboard, text=f"{start+i+1}. {visible[i][0]}: {visible[i][1]} pts").place(x=10, y=40 + i * 30)
    next = CTkButton(root, text="Next", corner_radius=16, border_color="#FFD700", border_width=2,
                     hover_color="#FFFFFF", bg_color="#fffcf2", command=lambda: [change(), home()])
    previous = CTkButton(root, text="Previous", corner_radius=16, border_color="#FFD700", border_width=2,
                         hover_color="#FFFFFF", bg_color="#fffcf2", command=lambda: [previous_page(), home()])
    upload_trash_found = CTkButton(root, text="increase points", corner_radius=16, border_color="#FFD700",
                                   border_width=2, hover_color="#FFFFFF", bg_color="#fffcf2", command=call_model)
    find_recycling = CTkButton(root, text="Recycling Centers", corner_radius=16, border_color="#FFD700",
                               border_width=2, hover_color="#FFFFFF", bg_color="#fffcf2", command=open_recycling_finder)
    previous.place(x=120, y=520)
    next.place(x=330, y=520)
    upload_trash_found.place(x=330, y=560)
    find_recycling.place(x=120, y=560)

def check():
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username_enter.get(), password_enter.get()))
    if cursor.fetchone():
        home()
    else:
        CTkLabel(root, text="Incorrect credentials", text_color="red").place(relx=0.5, rely=0.9, anchor="center")

def save_user():
    cursor.execute("INSERT INTO users (username, password, score) VALUES (?, ?, ?)", (username_enter.get(), password_enter.get(), 0))
    conn.commit()
    login()

def login():
    global username_enter, password_enter
    window0 = CTkFrame(root, width=600, height=600)
    window0.place(x=0, y=0)
    bg_image = PILImage.open("/Users/Mukhil/Desktop/Sonoma_Hacks/backdrop.png")
    bg_image = bg_image.resize((600, 600))
    bg_ctk_image = CTkImage(light_image=bg_image, dark_image=bg_image, size=(600, 600))
    bg_label = CTkLabel(window0, image=bg_ctk_image, text="")
    bg_label.place(x=0, y=0)
    window1 = CTkFrame(root, width=300, height=300, fg_color="#edf2f4", corner_radius=16, border_color="#2b2d42", border_width=3)
    window1.place(x=150, y=150)
    window1.pack_propagate(False)
    username_label = CTkLabel(window1, text="Enter your username:")
    password_label = CTkLabel(window1, text="Enter your password:")
    username_enter = CTkEntry(window1, placeholder_text="Enter username here:")
    password_enter = CTkEntry(window1, placeholder_text="Enter password here:", show="*")
    submit_button = CTkButton(window1, text="Submit", corner_radius=16, border_color="#FFD700", border_width=2,
                              hover_color="#FFFFFF", command=check)
    sign_in = CTkButton(window1, text="Create", corner_radius=16, border_color="#FFD700", border_width=2,
                        hover_color="#FFFFFF", command=signin)
    username_label.place(relx=0.5, rely=0.1, anchor="n")
    username_enter.place(relx=0.5, rely=0.2, anchor="n")
    password_label.place(relx=0.5, rely=0.35, anchor="n")
    password_enter.place(relx=0.5, rely=0.45, anchor="n")
    submit_button.place(relx=0.5, rely=0.65, anchor="n")
    sign_in.place(relx=0.5, rely=0.78, anchor="n")

def signin():
    global username_enter, password_enter
    window0 = CTkFrame(root, width=600, height=600)
    window0.place(x=0, y=0)
    bg_image = PILImage.open("/Users/Mukhil/Desktop/Sonoma_Hacks/backdrop.png")
    bg_image = bg_image.resize((600, 600))
    bg_ctk_image = CTkImage(light_image=bg_image, dark_image=bg_image, size=(600, 600))
    bg_label = CTkLabel(window0, image=bg_ctk_image, text="")
    bg_label.place(x=0, y=0)
    window1 = CTkFrame(root, width=300, height=300, fg_color="#edf2f4", corner_radius=16, border_color="#2b2d42", border_width=3)
    window1.place(x=150, y=150)
    window1.pack_propagate(False)
    username_label = CTkLabel(window1, text="Enter your username:")
    password_label = CTkLabel(window1, text="Enter your password:")
    username_enter = CTkEntry(window1, placeholder_text="Enter username here:")
    password_enter = CTkEntry(window1, placeholder_text="Enter password here:", show="*")
    submit_button = CTkButton(window1, text="Submit", corner_radius=16, border_color="#FFD700", border_width=2,
                              hover_color="#FFFFFF", command=save_user)
    log_in = CTkButton(window1, text="Login", corner_radius=16, border_color="#FFD700", border_width=2,
                       hover_color="#FFFFFF", command=login)
    username_label.place(relx=0.5, rely=0.1, anchor="n")
    username_enter.place(relx=0.5, rely=0.2, anchor="n")
    password_label.place(relx=0.5, rely=0.35, anchor="n")
    password_enter.place(relx=0.5, rely=0.45, anchor="n")
    submit_button.place(relx=0.5, rely=0.65, anchor="n")
    log_in.place(relx=0.5, rely=0.78, anchor="n")

login()
root.mainloop()
