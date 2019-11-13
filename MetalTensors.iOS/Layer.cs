using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Layer
    {
        static int nextId = 1;
        readonly string autoLabel;

        readonly ConcurrentDictionary<IntPtr, MPSNNFilterNode> deviceFilterNodes =
            new ConcurrentDictionary<IntPtr, MPSNNFilterNode> ();

        public abstract int InputCount { get; }

        public string Label => autoLabel;

        protected Layer ()
        {
            var id = Interlocked.Increment (ref nextId);
            autoLabel = GetType ().Name + id;
        }

        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public MPSNNImageNode GetMetalImageNode (Tensor[] inputs, IMTLDevice device)
        {
            var f = GetFilterNode (inputs, device);
            //Console.WriteLine (f.ResultImage.DebugDescription);
            return f.ResultImage;
        }

        public Tensor GetOutput (params Tensor[] inputs)
        {
            return new LayerOutputTensor (this, inputs);
        }

        public Task<Tensor> ExecuteAsync (Tensor[] inputs, IMTLDevice device)
        {
            if (inputs.Length < InputCount)
                throw new ArgumentException (nameof (inputs));

            var tcs = new TaskCompletionSource<Tensor> ();
            ThreadPool.QueueUserWorkItem (StartGraph);
            return tcs.Task;

            void StartGraph (object s)
            {
                try {
                    var node = GetFilterNode (inputs, device);

                    using var graph = new MPSNNGraph (device, node.ResultImage, true) {
                        Format = MPSImageFeatureChannelFormat.Float32,
                    };
                    //Console.WriteLine (graph.DebugDescription);

                    var sourceHandles = graph.SourceImageHandles;
                    var sources = sourceHandles.Select (x => ((TensorHandle)x).Tensor.GetMetalImage ()).ToArray ();

                    var r = graph.Execute (sources, (image, error) => {
                        if (error != null) {
                            tcs.SetException (new Exception (error.Description));
                        }
                        else {
                            var t = new MPSImageTensor (image);
                            tcs.SetResult (t);
                        }
                    });
                }
                catch (Exception ex) {
                    tcs.TrySetException (ex);
                }
            }
        }

        static IMTLDevice? FindDevice (Tensor[] tensors)
        {
            // TODO: Scan inputs for the correct device to use
            return null;
        }

        protected MPSNNFilterNode GetFilterNode (Tensor[] inputs, IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceFilterNodes.TryGetValue (key, out var node))
                return node;

            var inputImageNodes = inputs.Select (x => (x.GetMetalImageNode (device), x.Shape)).ToArray ();
            node = CreateFilterNode (inputImageNodes, device);
            deviceFilterNodes.TryAdd (key, node);
            return node;
        }

        protected abstract MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device);
    }
}
