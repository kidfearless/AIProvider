namespace AIProvider;

public record ChatModel(string Model);
public record AzureChatModel(string Model, string Deployment) : ChatModel(Model);