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
            Console.WriteLine("Server is starting...");
            byte[] data = new byte[1024];

                IPEndPoint ipep = new IPEndPoint(IPAddress.Any, 55555); ;

                Socket newsock = new Socket(AddressFamily.InterNetwork,
                                SocketType.Stream, ProtocolType.Tcp);

                newsock.Bind(ipep);
                newsock.Listen(10);
                Console.WriteLine("Waiting for a client...");

                Socket client = newsock.Accept();
                IPEndPoint newclient = (IPEndPoint)client.RemoteEndPoint;
                Console.WriteLine("Connected with {0} at port {1}",
                                newclient.Address, newclient.Port);

                while (true)
                {
                    Console.WriteLine("Recieving Data");
                    data = ReceiveVarData(client);
                    MemoryStream ms = new MemoryStream(data);
                    try
                    {
                        Console.WriteLine("Success!");
                        ImageConverter imageConverter = new System.Drawing.ImageConverter();
                        System.Drawing.Image image = imageConverter.ConvertFrom(data) as System.Drawing.Image;
                        Image bmp = Image.FromStream(ms);
                        pictureBox1.Image = bmp;
                        
                    }
                    catch (ArgumentException e)
                    {
                        Console.WriteLine("something broke");
                    }


                    if (data.Length == 0)
                        newsock.Listen(10);
                }
                //Console.WriteLine("Disconnected from {0}", newclient.Address);
                client.Close();
                newsock.Close();

            }

        

        private static byte[] ReceiveVarData(Socket s)
        {
            int total = 0;
            int recv;
            byte[] datasize = new byte[4];

            recv = s.Receive(datasize, 0, 4, 0);
            int size = BitConverter.ToInt32(datasize, 0);
            int dataleft = size;
            byte[] data = new byte[size];


            while (total < size)
            {
                recv = s.Receive(data, total, dataleft, 0);
                if (recv == 0)
                {
                    break;
                }
                total += recv;
                dataleft -= recv;
            }
            return data;
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
