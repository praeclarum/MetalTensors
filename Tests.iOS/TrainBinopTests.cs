using System;
using System.Collections.Generic;
using System.Linq;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class TrainBinopTests
    {
        [Test]
        public void And ()
        {
            TrainBinop ((a, b) => a && b);
        }

        void TrainBinop (Func<bool, bool, bool> binop)
        {
            var x = Tensor.Input ("x", 2);
            var y = x.Dense (16).ReLU ().Dense (1).ReLU ();

            var labels = Tensor.Labels ("y", 1);

            var loss = y.Loss (labels, LossType.MeanSquaredError);

            var rand = new Random ();

            var history = loss.Train (GenTrainingData);

            //var batch = history.Batches[^1];

            

            Console.WriteLine (history);
            foreach (var batch in history.Batches) {
                Console.WriteLine ("------");
                foreach (var l in batch.Loss) {
                    Console.WriteLine ($"Loss {l[0]}");
                }
            }

            IEnumerable<Tensor> GenTrainingData (TensorHandle[] handles)
            {
                var r = new Tensor[handles.Length];

                var x0 = rand.NextDouble ();
                var x1 = rand.NextDouble ();
                var x0b = x0 >= 0.5;
                var x1b = x1 >= 0.5;
                var yb = binop (x0b, x1b);
                var y0 = (yb ? 1.0 : 0.0) + (rand.NextDouble () - 0.5) * 0.001;
                //Console.WriteLine ($"{x0} ? {x1} = {y0}");

                for (var i = 0; i < handles.Length; i++) {
                    if (handles[i].Label == "x") {
                        r[i] = Tensor.Array (x0, x1);
                    }
                    else {
                        r[i] = Tensor.Array (y0);
                    }
                }
                return r;
            }
        }
    }
}
