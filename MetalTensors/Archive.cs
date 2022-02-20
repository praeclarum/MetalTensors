using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace MetalTensors
{
    public interface IHasBuffers
    {
        void ReadBuffers (ReadBuffer reader);
        void WriteBuffers (WriteBuffer writer);
    }

    public delegate float[]? ReadBuffer (string name);
    public delegate void WriteBuffer (string name, ReadOnlySpan<float> values);

    public class ArchiveReader
    {
        private readonly ZipArchive archive;

        public ArchiveReader (Stream stream)
        {
            archive = new ZipArchive (stream, ZipArchiveMode.Read, leaveOpen: true, Encoding.UTF8);
        }

        public T Read<T> () where T : Configurable
        {
            var entryName = $"{typeof (T).Name}.root.xml";
            var entry = archive.Entries.FirstOrDefault (x => x.FullName == entryName);
            if (entry == null)
                throw new Exception ($"Failed to find root entry");
            T value;
            var references = new Dictionary<int, Configurable> ();
            using (var s = entry.Open ()) {
                value = Config.Read<T> (s, references);
            }
            foreach (var r in references) {
                var c = r.Value;
                var oid = r.Key;
                if (c is IHasBuffers hb) {
                    hb.ReadBuffers ((x) => ReadBuffer (c, oid, x));
                }
            }
            return value;
        }

        float[]? ReadBuffer (Configurable bufferOwner, int originalId, string bufferName)
        {
            var entryName = $"{bufferOwner.GetType ().Name}.{originalId}.{bufferName}.floats";
            var entry = archive.GetEntry (entryName);
            if (entry == null)
                return null;
            var numBytes = entry.Length;
            var numFloats = numBytes / 4;
            var result = new float[numFloats];
            var bspan = MemoryMarshal.Cast<float, byte> (result);
            using (var s = entry.Open ()) {
                s.Read (bspan);
            }
            return result;
        }
    }

    public class ArchiveWriter : IDisposable
    {
        bool disposed = false;
        readonly HashSet<string> wroteBuffers = new HashSet<string> ();
        readonly ZipArchive archive;

        public ArchiveWriter (Stream stream)
        {
            archive = new ZipArchive (stream, ZipArchiveMode.Create, leaveOpen: true, Encoding.UTF8);
        }

        public void Dispose ()
        {
            if (!disposed) {
                disposed = true;
                archive.Dispose ();
            }
        }

        public void Write (Configurable root)
        {
            if (disposed)
                throw new ObjectDisposedException (nameof (ArchiveWriter));
            var config = root.Config;
            var entryName = $"{config.ObjectType}.root.xml";
            var entry = archive.CreateEntry (entryName);
            var references = new HashSet<Configurable> ();
            using (var s = entry.Open ()) {
                config.Write (s, references);
            }
            references.Add (root);
            foreach (var r in references) {
                if (r is IHasBuffers hb) {
                    var c = r;
                    hb.WriteBuffers ((x, y) => WriteBuffer (c, x, y));
                }
            }
        }

        void WriteBuffer (Configurable source, string bufferName, ReadOnlySpan<float> values)
        {
            var entryName = $"{source.GetType().Name}.{source.Id}.{bufferName}.floats";
            if (wroteBuffers.Contains (entryName))
                return;
            wroteBuffers.Add (entryName);
            var entry = archive.CreateEntry (entryName);
            using var s = entry.Open ();
            s.Write (MemoryMarshal.Cast<float, byte> (values));
        }
    }
}
