using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Net;
using System.Net.Sockets;
using System.IO;
using System.Threading;
using System.Windows.Media.Imaging;
using System.Drawing.Imaging;
using System.Windows;
using System.Windows.Media;
using System.Windows.Forms.VisualStyles;
using Windows.Storage.Streams;
using System.Diagnostics.Tracing;
using System.Net.WebSockets;
using Windows.System;

namespace SeniorDesignProject3._0
{
    public partial class Form1 : Form
    {

        public Form1()
        {
            InitializeComponent();
        }
        void pictureBox1_Click(object sender, EventArgs e) { }
        Process proc = new Process();

        public void DTW()
        {
            string progToRun = @"C:\Users\david\Desktop\DTWpipe\main.py";
            //string progToRun = @"C:\Users\david\Desktop\Test\test.py";
            proc.StartInfo.FileName = "python.exe";
            proc.StartInfo.RedirectStandardOutput = true;
            proc.StartInfo.UseShellExecute = false;
            proc.StartInfo.CreateNoWindow = true;

            // call hello.py to concatenate passed parameters
            proc.StartInfo.Arguments = string.Concat(progToRun);
            proc.Start();
        }

        public void getFrames()
        {
            //create a TCP socket
            Console.WriteLine("Creating the socket...");
            Socket socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            IPAddress ipAddress = IPAddress.Parse("192.168.56.1");
            IPEndPoint localEndPoint = new IPEndPoint(ipAddress, 1234);
            socket.Bind(localEndPoint);
            socket.Listen(1);
            Socket clientSocket = socket.Accept();
            Console.WriteLine("Connected!");

            while (true)
            {
                //Console.WriteLine("Trying..");
                byte[] buffer = new byte[1000000];
                int bytesRead = clientSocket.Receive(buffer);

                while (bytesRead != 0)
                {
                    //Console.WriteLine("Working...");
                    byte[] imageData = new byte[bytesRead];
                    Array.Copy(buffer, imageData, bytesRead);
                    using (MemoryStream ms = new MemoryStream(imageData))
                    {
                            Image image = Image.FromStream(ms);
                            pictureBox1.Image = image;
                            //Console.WriteLine("Image being Display..");
                            break;
               
                    }

                }

            }

        }

        // Enable Recording
        void button1_Click(object sender, EventArgs e)
        {
            Task task1 = Task.Factory.StartNew(() => getFrames());
            Task task2 = Task.Factory.StartNew(() => DTW());
        }

        // Start Translating
        private void button2_Click(object sender, EventArgs e)
        { 
            Console.WriteLine("Translating in progress...");
            using (StreamWriter writer = new StreamWriter("C:\\Users\\david\\Desktop\\DTWpipe\\action.txt"))
            {
                writer.WriteLine("R");
            }

            Task task3 = Task.Factory.StartNew(() => UpdateTextBox());
    
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Console.WriteLine("Exiting...");
            using (StreamWriter writer = new StreamWriter("C:\\Users\\david\\Desktop\\DTWpipe\\action.txt"))
            {
                writer.WriteLine("Q");
            }
            proc.Kill();
        }

        // Updating the TextBox
        private void UpdateTextBox()
        {
            string text;
            Thread.Sleep(5000);
        
            using (StreamReader reader = new StreamReader("C:\\Users\\david\\Desktop\\DTWpipe\\translated_text.txt"))
            {
                text = reader.ReadLine();

                    textBox1.Invoke(new MethodInvoker(delegate { textBox1.Text = text; }));
                    Console.WriteLine(text);
                    Console.WriteLine("Done!");


            }

        }
    }
}