using Microsoft.Extensions.AI;

namespace AIProvider.Messages;

public record UserMessage(string Content) : Message(Content)
{
    public override string Role { get; set; } = "user";
    public List<AIContent> Files { get; set; } = [];

    public UserMessage WithFiles(params DataContent[] content)
    {
        return this with { Files = [.. this.Files, .. content] };
    }

    public UserMessage WithFile(DataContent content)
    {
        return this with { Files = [content] };
    }
}