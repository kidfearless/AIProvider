using System.Text;

namespace AIProvider;


public abstract partial record Provider
{
    public record ChatSession(Provider Provider, ChatModel ChatModel)
    {
        public List<Messages.Message> Messages { get; set; } = [];
        public int ShortTermMemoryLength { get; set; } = 20;
        public ulong? MaxOutputTokens { get; set; }

        public IAsyncEnumerable<Response> StreamResponseAsync(CancellationToken cancellationToken = default) => Provider.StreamResponseAsync(this, cancellationToken);
        public async Task<T> StructuredOutputAsync<T>(CancellationToken cancellationToken = default) => await Provider.StructuredOutputAsync<T>(this);

        public virtual async Task<Response> GetResponseAsync()
        {
            var builder = new StringBuilder(2048);
            await foreach (var res in StreamResponseAsync())
            {
                builder.Append(res.Content);
            }

            return new Response(builder.ToString());
        }
    }

}