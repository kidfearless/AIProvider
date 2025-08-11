using System;

using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;

using static System.Xml.Schema.XmlSchemaInference;
using static AIProvider.Provider;

namespace AIProvider;

public abstract partial record Provider : IDisposable
{
    protected abstract string Key { get; }
    protected abstract string Url { get; }

    protected string? ApiKey { get; set; }
    protected bool IsInitialized { get; set; }
    public List<AITool> Tools { get; private set; } = [];
    protected IConfiguration? Configuration { get; set; }


    public virtual ChatSession CreateChatSession(ChatModel chatModel) => new ChatSession(this, chatModel);
    public abstract Task<List<ChatModel>> GetModelsAsync();

    protected abstract IAsyncEnumerable<Response> StreamResponseAsync(ChatSession session, CancellationToken cancellationToken);
    protected abstract Task<T> StructuredOutputAsync<T>(ChatSession session);

    public virtual void Initialize(string apiKey, IConfiguration? options = null)
    {
        ApiKey = apiKey;
        Configuration = options;

        IsInitialized = true;
    }

    public static Provider GetProvider(string key, string apiKey, IConfiguration? options = null)
    {
        Provider provider = key switch
        {
            "OpenAI" => new OpenAiProvider(),
            "Anthropic" => new AnthropicProvider(),
            "Gemini" => new GeminiProvider(),
            "AzureOpenAI" => new AzureProvider(),
            _ => throw new Exception("Invalid provider")
        };

        provider.Initialize(apiKey, options);
        return provider;
    }

    public virtual void LoadTools(List<AITool> tools) => Tools = tools;

    public void Dispose()
    {
        throw new NotImplementedException();
    }

}