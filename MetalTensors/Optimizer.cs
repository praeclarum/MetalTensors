using System;

namespace MetalTensors
{
    public abstract class Optimizer
    {
        public const float DefaultLearningRate = 1e-3f;

        public float LearningRate = DefaultLearningRate;
    }

    public class AdamOptimizer : Optimizer
    {
        public AdamOptimizer (float learningRate = DefaultLearningRate)
        {
            LearningRate = learningRate;
        }
    }
}
