using AIProvider.Messages;

using System.Text;

namespace AIProvider;

public record ChatSession
{
    private readonly Func<ChatSession, CancellationToken, IAsyncEnumerable<Response>> streamAction;

    public List<Message> Messages { get; set; } = [];
    public int ShortTermMemoryLength { get; set; } = 20;
    public Provider Provider { get; }
    public ChatModel ChatModel { get; }

    public ChatSession(Provider provider, ChatModel chatModel, Func<ChatSession, CancellationToken, IAsyncEnumerable<Response>> streamAction)
    {
        Provider = provider;
        ChatModel = chatModel;
        this.streamAction = streamAction;
    }

    public IAsyncEnumerable<Response> StreamResponseAsync(CancellationToken cancellationToken = default) => streamAction(this, cancellationToken);

    public async Task<Response> GetResponseAsync()
    {
        var builder = new StringBuilder(2048);
        await foreach (var res in StreamResponseAsync())
        {
            builder.Append(res.Content);
        }

        return new Response(builder.ToString());
    }
}