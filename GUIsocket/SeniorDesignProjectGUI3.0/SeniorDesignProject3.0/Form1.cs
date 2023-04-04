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
            string fileName = "C:\\Users\\david\\Desktop\\DTWpipe\\main.py";
            Process p = new Process();
            p.StartInfo = new ProcessStartInfo("Python.exe", fileName)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                CreateNoWindow = true
            };

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Thread t = new Thread(new ThreadStart(DTW));

            IPEndPoint iep = new IPEndPoint(IPAddress.Parse("127.0.0.1"), 6008);
            using (Socket client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                client.Connect(iep);

                    string input = Console.ReadLine();

                        // receive data
                        byte[] buffer = new byte[1000000];
                        client.Receive(buffer, buffer.Length, SocketFlags.None);
                        Console.WriteLine("Receive success");

                        File.WriteAllBytes("1.jpg", buffer);

                    Image image = Image.FromFile("1.jpg");
                    pictureBox1.Image = image;

                
            }

        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {
            

        }
    }
}
