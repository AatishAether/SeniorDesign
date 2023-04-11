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

namespace SeniorDesignProject3._0
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        public void DTW()
        {
            string progToRun = @"C:\Users\david\Desktop\DTWpipe\main.py";
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
            Bitmap MyImage;
            
            Console.WriteLine("Here 1");
            IPEndPoint iep = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5555);
               using (Socket client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
                {
                    client.Connect(iep);

                    // receive data
                    byte[] buffer = new byte[1000000];
                    while (true) { 
                        client.Receive(buffer, buffer.Length, SocketFlags.None);
                        Console.WriteLine("Receive success");

                    //File.WriteAllBytes("C:\\users\\david\\desktop\\1.jpg", buffer);
                    //Image image = Image.FromFile("C:\\users\\david\\desktop\\1.jpg");

                    try
                    {

                        byte[] oByteArray = File.ReadAllBytes("C:\\Users\\david\\Desktop\\DTWpipe\\data.bin");
                        Image image = (Image)(BitmapImage2Bitmap(ConvertToImage(oByteArray)));

                        pictureBox1.Image = image;

                    } catch(NotSupportedException)
                    {



                    }


                }
            }
        }

        private Bitmap BitmapImage2Bitmap(BitmapImage bitmapImage)
        {
            // BitmapImage bitmapImage = new BitmapImage(new Uri("../Images/test.png", UriKind.Relative));

            using (MemoryStream outStream = new MemoryStream())
            {
                BitmapEncoder enc = new BmpBitmapEncoder();
                enc.Frames.Add(BitmapFrame.Create(bitmapImage));
                enc.Save(outStream);
                System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap(outStream);

                return new Bitmap(bitmap);
            }
        }

        public BitmapImage ConvertToImage(byte[] Binary)
        {

            byte[] buffer = Binary.ToArray();
            MemoryStream stream = new MemoryStream(buffer);
            BitmapImage image = new BitmapImage();
            image.BeginInit();
            image.StreamSource = stream;
            image.EndInit();
            return image;
        }

        void button1_Click(object sender, EventArgs e)
            {
                Task task1 = Task.Factory.StartNew(() => DTW());
                Thread.Sleep(10000);
                Task task2 = Task.Factory.StartNew(() => getFrames());

            }

            void pictureBox1_Click(object sender, EventArgs e)
            {


            }
        }
    }
