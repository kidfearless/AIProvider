using Microsoft.Extensions.AI;

namespace AIProvider.Messages;

public record UserMessage(string Content) : Message(Content)
{
    public override string Role { get; set; } = "user";
    public List<AIContent> Files { get; set; } = [];
}