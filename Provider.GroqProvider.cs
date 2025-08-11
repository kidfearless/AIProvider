using System.Runtime.CompilerServices;

using AIProvider.Messages;

using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;

namespace AIProvider;

public abstract partial record Provider
{
  public record GroqProvider : OpenAiProvider
  {
    private string _url = "https://api.groq.com/openai/v1/";

    protected override string Key => "Groq";
    protected override string Url => _url;
    protected override OpenAI.OpenAIClientOptions Options => new() { Endpoint = new(Url) };

    public override void Initialize(string apiKey, IConfiguration? options = null)
    {
      base.Initialize(apiKey, options);
      if (options != null)
      {
        var url = options[$"Provider:{Key}:Url"];
        if (url is not null or "")
        {
          _url = url;
        }
      }
    }

    public record GroqChatSession(GroqProvider GroqProvider, ChatModel ChatModel) : Provider.ChatSession(GroqProvider, ChatModel)
    {
      public string? ServiceTier { get; set; } = "auto";
      public bool? IncludeReasoning { get; set; }
      public string? ReasoningFormat { get; set; }
      public string? ReasoningEffort { get; set; }

      public override Task<Response> GetResponseAsync()
      {
        return GroqProvider.GetResponseAsync(this);
      }
    }

    public override ChatSession CreateChatSession(ChatModel chatModel)
    {
      return new GroqChatSession(this, chatModel);
    }

    protected virtual async Task<Response> GetResponseAsync(GroqChatSession session)
    {
      if (!IsInitialized)
      {
        throw new Exception("Provider not initialized");
      }

      if (!session.Messages.Any())
      {
        throw new Exception("No messages to send");
      }

      using var chatClient = new OpenAI.OpenAIClient(Auth, Options)
          .GetChatClient(session.ChatModel.Model)
          .AsIChatClient()
          .AsBuilder()
          .UseFunctionInvocation()
          .Build();

      var messages = session.Messages.Select(m =>
      {
        return m switch
        {
          AssistantMessage a => ConvertToChatMessage(a),
          SystemPromptMessage a => ConvertToChatMessage(a),
          UserMessage a => ConvertToChatMessage(a),
          Messages.Message a => ConvertToChatMessage(a),
          _ => throw new NotImplementedException()
        };
      })
      .TakeLast(session.ShortTermMemoryLength + 1)
      .ToList();

      var chatOptions = new ChatOptions()
      {
        Tools = Tools,
        MaxOutputTokens = (int?)session.MaxOutputTokens,
        AdditionalProperties = BuildGroqSpecificOptions(session)
      };

      var response = await chatClient.GetResponseAsync(messages, chatOptions);
      return new Response(response.Text ?? string.Empty);
    }

    protected override async IAsyncEnumerable<Response> StreamResponseAsync(ChatSession session, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
      if (!IsInitialized)
      {
        throw new Exception("Provider not initialized");
      }

      if (!session.Messages.Any())
      {
        throw new Exception("No messages to send");
      }

      var groqSession = session as GroqChatSession;

      using var chatClient = new OpenAI.OpenAIClient(Auth, Options)
          .GetChatClient(session.ChatModel.Model)
          .AsIChatClient()
          .AsBuilder()
          .UseFunctionInvocation()
          .Build();

      var messages = session.Messages.Select(m =>
      {
        return m switch
        {
          AssistantMessage a => ConvertToChatMessage(a),
          SystemPromptMessage a => ConvertToChatMessage(a),
          UserMessage a => ConvertToChatMessage(a),
          Messages.Message a => ConvertToChatMessage(a),
          _ => throw new NotImplementedException()
        };
      })
      .TakeLast(session.ShortTermMemoryLength + 1)
      .ToList();

      var chatOptions = new ChatOptions()
      {
        Tools = Tools,
        MaxOutputTokens = (int?)session.MaxOutputTokens,
        AdditionalProperties = groqSession != null ? BuildGroqSpecificOptions(groqSession) : null
      };

      var stream = chatClient.GetStreamingResponseAsync(messages, chatOptions, cancellationToken: cancellationToken);
      await foreach (var message in stream)
      {
        foreach (var contentPart in message.Contents.OfType<TextContent>())
        {
          if (contentPart.Text != null)
          {
            yield return new(contentPart.Text);
          }
        }
      }
    }

    private AdditionalPropertiesDictionary? BuildGroqSpecificOptions(GroqChatSession session)
    {
      var options = new AdditionalPropertiesDictionary();

      if (session.ServiceTier != null)
      {
        options["service_tier"] = session.ServiceTier;
      }

      if (session.IncludeReasoning.HasValue)
      {
        options["include_reasoning"] = session.IncludeReasoning.Value;
      }

      if (session.ReasoningFormat != null)
      {
        options["reasoning_format"] = session.ReasoningFormat;
      }

      if (session.ReasoningEffort != null)
      {
        options["reasoning_effort"] = session.ReasoningEffort;
      }

      return options.Count > 0 ? options : null;
    }

    public override async Task<List<ChatModel>> GetModelsAsync()
    {
      if (!IsInitialized)
      {
        throw new Exception("Provider not initialized");
      }

      var client = new OpenAI.OpenAIClient(Auth, Options)
          .GetOpenAIModelClient();
      var models = await client.GetModelsAsync();
      return models.Value.Select(m => new ChatModel(m.Id)).ToList();
    }
  }
}
