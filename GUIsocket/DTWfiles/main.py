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
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.Windows;
using System.Windows.Media;
using System.Windows.Forms.VisualStyles;
using Windows.Storage.Streams;
using System.Diagnostics.Tracing;
using System.Net.WebSockets;

namespace SeniorDesignProject3._0
{
    public partial class Form1 : Form
    {
        ArgumentException ex;

        public Form1()
        {
            InitializeComponent();
        }

        public void DTW()
        {
            string progToRun = @"C:\Users\david\Desktop\DTWpipe\main.py";
            //string progToRun = @"C:\Users\david\Desktop\Test\test.py";
            Process proc = new Process();
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
            IPAddress ipAddress = IPAddress.Parse("172.16.206.245");
            IPEndPoint localEndPoint = new IPEndPoint(ipAddress, 1234);
            socket.Bind(localEndPoint);
            socket.Listen(1);
            Socket clientSocket = socket.Accept();
            Console.WriteLine("Connected!");

            while (true)
            {
                Console.WriteLine("Trying..");
                byte[] buffer = new byte[1000000];
                Console.WriteLine("I am");
                int bytesRead = clientSocket.Receive(buffer);
                while (bytesRead != 0)
                {
                    Console.WriteLine("Working...");
                    byte[] imageData = new byte[bytesRead];
                    Array.Copy(buffer, imageData, bytesRead);
                    using (MemoryStream ms = new MemoryStream(imageData))
                    {
                        try
                        {
                            Image image = Image.FromStream(ms);
                            pictureBox1.Image = image;
                            Console.WriteLine("Image being Display..");
                            break;
                        }
                        catch (ArgumentException ex)
                        {
                            Console.WriteLine(ex.Message);
                        }

                    }

                }

            }

        }


        void button1_Click(object sender, EventArgs e)
        {


            Task task2 = Task.Factory.StartNew(() => getFrames());
            Thread.Sleep(1000);
            Task task1 = Task.Factory.StartNew(() => DTW());




        }

        void pictureBox1_Click(object sender, EventArgs e)
        {


        }
    }
}
