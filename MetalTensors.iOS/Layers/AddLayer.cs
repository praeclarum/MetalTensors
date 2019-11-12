using System;
using System.Linq;
using System.Threading.Tasks;
using Foundation;
using MetalPerformanceShaders;
using MetalTensors.Tensors;

namespace MetalTensors.Layers
{
    public class AddLayer : Layer
    {
        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        public override Tensor Compute (Tensor[] inputs)
        {
            if (inputs.Length != 2)
                throw new ArgumentException (nameof (inputs));

            var inputImageNodes = inputs.Select (x => x.GetImageNode ()).ToArray ();
            var node = new MPSNNAdditionNode (inputImageNodes);

            var device = MetalExtensions.Current (null);
            using var graph = new MPSNNGraph (device, node.ResultImage, true) {
                Format = MPSImageFeatureChannelFormat.Float32,
            };
            //Console.WriteLine (graph.DebugDescription);

            var sourceHandles = graph.SourceImageHandles;
            var sources = sourceHandles.Select (x => ((TensorHandle)x).Tensor.GetImage ()).ToArray ();

            var tcs = new TaskCompletionSource<NSError?> ();
            var result = graph.Execute (sources, (image, error) => {
                tcs.SetResult (error);
            });
            var exeError = tcs.Task.Result;
            exeError.ValidateNoError ();

            return new MPSImageTensor (result);
        }
    }
}
