import Tkinter as tk
import PIL
from PIL import Image,ImageTk
from tkFileDialog import askopenfilename
import tensorflow as tf, sys
filename = askopenfilename(initialdir="",#Enter the path of file to be chosen 
                           filetypes =(("Image File", "*.jpg"),("All Files","*.*")),
                           title = "Choose a file."
                           )
image_path = filename
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("")]#Enter the path of output_labels.txt
with tf.gfile.FastGFile("", 'rb') as f:#in "" Enter the path of  output_graph.pb
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for i in top_k:
       human_string = label_lines[i]
       score = predictions[0][i]
        #print('{} (score = {:.2f} %)'.format(human_string, score*100))
       root=tk.Toplevel()
       root.title('Gender')
       root.geometry("1500x1500")
       root.configure(background='black')
       im=Image.open(image_path)
       ph = ImageTk.PhotoImage(im)
       panel=tk.Label(root,image=ph,text=human_string,compound="bottom")
       panel.pack(side="bottom",fill="both",expand="yes")
       root.mainloop()
       break
        
