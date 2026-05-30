---
title: 行为型设计模式
icon: puzzle-piece
order: 3
---

# Java 行为型设计模式详解

行为型设计模式关注的是对象之间的职责分配和通信方式。

---

## 一、观察者模式（Observer Pattern）

### 1.1 模式定义

定义对象间的一对多依赖关系，当一个对象改变状态时，它的所有依赖者都会收到通知并自动更新。

### 1.2 Java 原生支持

```java
// JDK 内置的观察者模式
public class NewsAgency extends Observable {
    private String news;
    
    public void setNews(String news) {
        this.news = news;
        setChanged(); // 标记状态已改变
        notifyObservers(news); // 通知所有观察者
    }
}

public class NewsChannel implements Observer {
    private String news;
    
    @Override
    public void update(Observable o, Object arg) {
        this.news = (String) arg;
        System.out.println("收到新闻: " + news);
    }
}

// 使用
NewsAgency agency = new NewsAgency();
NewsChannel channel = new NewsChannel();
agency.addObserver(channel);
agency.setNews("今日股市大涨");
```

### 1.3 Spring 事件驱动

```java
// 自定义事件
public class OrderCreatedEvent extends ApplicationEvent {
    private final String orderId;
    private final double amount;
    
    public OrderCreatedEvent(Object source, String orderId, double amount) {
        super(source);
        this.orderId = orderId;
        this.amount = amount;
    }
    
    public String getOrderId() { return orderId; }
    public double getAmount() { return amount; }
}

// 事件监听器
@Component
public class OrderEventListener {
    
    @EventListener
    public void handleOrderCreated(OrderCreatedEvent event) {
        System.out.println("订单创建: " + event.getOrderId());
        // 发送通知、更新库存等
    }
    
    @EventListener
    @Async
    public void handleOrderCreatedAsync(OrderCreatedEvent event) {
        // 异步处理
        System.out.println("异步处理订单: " + event.getOrderId());
    }
}

// 发布事件
@Service
public class OrderService {
    @Autowired
    private ApplicationEventPublisher eventPublisher;
    
    public void createOrder(String orderId, double amount) {
        // 创建订单逻辑...
        
        // 发布事件
        eventPublisher.publishEvent(
            new OrderCreatedEvent(this, orderId, amount)
        );
    }
}
```

---

## 二、策略模式（Strategy Pattern）

### 2.1 模式定义

定义一系列算法，将每个算法封装起来，并使它们可以互相替换。

### 2.2 实战：促销策略

```java
// 策略接口
public interface DiscountStrategy {
    double calculate(double originalPrice);
}

// 具体策略 - 无折扣
public class NoDiscountStrategy implements DiscountStrategy {
    @Override
    public double calculate(double originalPrice) {
        return originalPrice;
    }
}

// 具体策略 - 满减
public class FullReductionStrategy implements DiscountStrategy {
    private double threshold;
    private double reduction;
    
    public FullReductionStrategy(double threshold, double reduction) {
        this.threshold = threshold;
        this.reduction = reduction;
    }
    
    @Override
    public double calculate(double originalPrice) {
        if (originalPrice >= threshold) {
            return originalPrice - reduction;
        }
        return originalPrice;
    }
}

// 具体策略 - 百分比折扣
public class PercentageDiscountStrategy implements DiscountStrategy {
    private double discountRate;
    
    public PercentageDiscountStrategy(double discountRate) {
        this.discountRate = discountRate;
    }
    
    @Override
    public double calculate(double originalPrice) {
        return originalPrice * (1 - discountRate);
    }
}

// 上下文
public class PriceCalculator {
    private DiscountStrategy strategy;
    
    public PriceCalculator(DiscountStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void setStrategy(DiscountStrategy strategy) {
        this.strategy = strategy;
    }
    
    public double calculatePrice(double originalPrice) {
        return strategy.calculate(originalPrice);
    }
}

// 使用示例
PriceCalculator calculator = new PriceCalculator(new NoDiscountStrategy());
System.out.println("原价: " + calculator.calculatePrice(100)); // 100

calculator.setStrategy(new FullReductionStrategy(100, 20));
System.out.println("满减: " + calculator.calculatePrice(150)); // 130

calculator.setStrategy(new PercentageDiscountStrategy(0.2));
System.out.println("八折: " + calculator.calculatePrice(100)); // 80
```

### 2.3 Spring 中的策略模式

```java
@Service
public class PaymentStrategyFactory {
    private final Map<PayType, PaymentStrategy> strategies;
    
    // Spring 自动注入所有 PaymentStrategy 实现
    public PaymentStrategyFactory(List<PaymentStrategy> strategyList) {
        strategies = strategyList.stream()
            .collect(Collectors.toMap(PaymentStrategy::getType, s -> s));
    }
    
    public PaymentStrategy getStrategy(PayType type) {
        return strategies.get(type);
    }
}
```

---

## 三、模板方法模式（Template Method Pattern）

### 3.1 模式定义

定义一个操作中的算法骨架，将某些步骤延迟到子类中实现。

### 3.2 实战：数据处理流程

```java
// 抽象模板
public abstract class DataProcessor {
    
    // 模板方法，定义算法骨架
    public final void process(String inputFile, String outputFile) {
        // 1. 读取数据
        List<String> data = readData(inputFile);
        
        // 2. 验证数据
        if (!validateData(data)) {
            throw new IllegalArgumentException("数据验证失败");
        }
        
        // 3. 转换数据（子类实现）
        List<String> transformed = transformData(data);
        
        // 4. 写入数据
        writeData(transformed, outputFile);
        
        // 5. 记录日志（可选钩子）
        logProcess(inputFile, outputFile, transformed.size());
    }
    
    // 具体方法
    protected List<String> readData(String inputFile) {
        System.out.println("读取文件: " + inputFile);
        return Arrays.asList("data1", "data2", "data3");
    }
    
    protected boolean validateData(List<String> data) {
        return data != null && !data.isEmpty();
    }
    
    protected void writeData(List<String> data, String outputFile) {
        System.out.println("写入文件: " + outputFile + ", 共 " + data.size() + " 条");
    }
    
    // 抽象方法，子类必须实现
    protected abstract List<String> transformData(List<String> data);
    
    // 钩子方法，子类可选覆盖
    protected void logProcess(String input, String output, int count) {
        // 默认空实现
    }
}

// 具体实现 - CSV 处理
public class CsvDataProcessor extends DataProcessor {
    @Override
    protected List<String> transformData(List<String> data) {
        return data.stream()
            .map(s -> s.toUpperCase())
            .collect(Collectors.toList());
    }
    
    @Override
    protected void logProcess(String input, String output, int count) {
        System.out.println("CSV 处理完成: " + count + " 条记录");
    }
}

// 具体实现 - JSON 处理
public class JsonDataProcessor extends DataProcessor {
    @Override
    protected List<String> transformData(List<String> data) {
        return data.stream()
            .map(s -> "{\"value\": \"" + s + "\"}")
            .collect(Collectors.toList());
    }
}

// 使用
DataProcessor csvProcessor = new CsvDataProcessor();
csvProcessor.process("input.csv", "output.csv");
```

### 3.3 Spring 中的模板方法

```java
// JdbcTemplate 是模板方法的典型应用
jdbcTemplate.query("SELECT * FROM users", new RowMapper<User>() {
    @Override
    public User mapRow(ResultSet rs, int rowNum) throws SQLException {
        // 只需要实现映射逻辑
        User user = new User();
        user.setId(rs.getLong("id"));
        user.setName(rs.getString("name"));
        return user;
    }
});
```

---

## 四、责任链模式（Chain of Responsibility Pattern）

### 4.1 模式定义

使多个对象都有机会处理请求，从而避免请求的发送者和接收者之间的耦合关系。

### 4.2 实战：审批流程

```java
// 抽象处理器
public abstract class Approver {
    protected Approver next;
    
    public Approver setNext(Approver next) {
        this.next = next;
        return next;
    }
    
    public abstract void approve(Request request);
}

// 具体处理器 - 项目经理
public class ProjectManager extends Approver {
    @Override
    public void approve(Request request) {
        if (request.getAmount() <= 10000) {
            System.out.println("项目经理审批通过: " + request.getAmount());
        } else if (next != null) {
            System.out.println("项目经理转交上级审批");
            next.approve(request);
        }
    }
}

// 具体处理器 - 部门经理
public class DepartmentManager extends Approver {
    @Override
    public void approve(Request request) {
        if (request.getAmount() <= 50000) {
            System.out.println("部门经理审批通过: " + request.getAmount());
        } else if (next != null) {
            System.out.println("部门经理转交上级审批");
            next.approve(request);
        }
    }
}

// 具体处理器 - 总经理
public class GeneralManager extends Approver {
    @Override
    public void approve(Request request) {
        if (request.getAmount() <= 100000) {
            System.out.println("总经理审批通过: " + request.getAmount());
        } else {
            System.out.println("金额超限，需要董事会审批");
        }
    }
}

// 请求对象
public class Request {
    private String type;
    private double amount;
    private String description;
    
    // getter/setter...
}

// 使用示例
Approver pm = new ProjectManager();
Approver dm = new DepartmentManager();
Approver gm = new GeneralManager();

pm.setNext(dm).setNext(gm);

Request request = new Request("采购", 35000, "购买服务器");
pm.approve(request);
```

### 4.3 Spring MVC 拦截器

```java
@Component
public class AuthInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, 
                           HttpServletResponse response, 
                           Object handler) {
        String token = request.getHeader("Authorization");
        if (!validateToken(token)) {
            response.setStatus(401);
            return false;
        }
        return true;
    }
}

@Component
public class LogInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, 
                           HttpServletResponse response, 
                           Object handler) {
        System.out.println("请求: " + request.getRequestURI());
        return true;
    }
}

@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Autowired
    private AuthInterceptor authInterceptor;
    
    @Autowired
    private LogInterceptor logInterceptor;
    
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(logInterceptor)
            .addInterceptor(authInterceptor); // 形成责任链
    }
}
```

---

## 五、命令模式（Command Pattern）

### 5.1 模式定义

将请求封装为对象，从而使你可以用不同的请求对客户进行参数化。

### 5.2 实战：遥控器

```java
// 命令接口
public interface Command {
    void execute();
    void undo();
}

// 接收者 - 电灯
public class Light {
    public void on() { System.out.println("灯开了"); }
    public void off() { System.out.println("灯关了"); }
}

// 接收者 - 电视
public class TV {
    public void on() { System.out.println("电视开了"); }
    public void off() { System.out.println("电视关了"); }
}

// 具体命令 - 开灯
public class LightOnCommand implements Command {
    private Light light;
    
    public LightOnCommand(Light light) {
        this.light = light;
    }
    
    @Override
    public void execute() {
        light.on();
    }
    
    @Override
    public void undo() {
        light.off();
    }
}

// 具体命令 - 开电视
public class TVOnCommand implements Command {
    private TV tv;
    
    public TVOnCommand(TV tv) {
        this.tv = tv;
    }
    
    @Override
    public void execute() {
        tv.on();
    }
    
    @Override
    public void undo() {
        tv.off();
    }
}

// 调用者 - 遥控器
public class RemoteControl {
    private Command[] onCommands;
    private Command[] offCommands;
    private Command undoCommand;
    
    public RemoteControl() {
        onCommands = new Command[7];
        offCommands = new Command[7];
        Command noCommand = new NoCommand();
        for (int i = 0; i < 7; i++) {
            onCommands[i] = noCommand;
            offCommands[i] = noCommand;
        }
        undoCommand = noCommand;
    }
    
    public void setCommand(int slot, Command onCmd, Command offCmd) {
        onCommands[slot] = onCmd;
        offCommands[slot] = offCmd;
    }
    
    public void onButtonPressed(int slot) {
        onCommands[slot].execute();
        undoCommand = onCommands[slot];
    }
    
    public void offButtonPressed(int slot) {
        offCommands[slot].execute();
        undoCommand = offCommands[slot];
    }
    
    public void undoButtonPressed() {
        undoCommand.undo();
    }
}

// 使用
RemoteControl remote = new RemoteControl();
Light light = new Light();
TV tv = new TV();

remote.setCommand(0, new LightOnCommand(light), new LightOffCommand(light));
remote.setCommand(1, new TVOnCommand(tv), new TVOffCommand(tv));

remote.onButtonPressed(0); // 开灯
remote.onButtonPressed(1); // 开电视
remote.undoButtonPressed(); // 关电视
```

---

## 六、状态模式（State Pattern）

### 6.1 模式定义

允许对象在其内部状态改变时改变它的行为。

### 6.2 实战：订单状态

```java
// 状态接口
public interface OrderState {
    void pay(Order order);
    void ship(Order order);
    void receive(Order order);
    void cancel(Order order);
    String getStateName();
}

// 具体状态 - 待支付
public class PendingState implements OrderState {
    @Override
    public void pay(Order order) {
        System.out.println("支付成功");
        order.setState(new PaidState());
    }
    
    @Override
    public void ship(Order order) {
        System.out.println("订单未支付，无法发货");
    }
    
    @Override
    public void receive(Order order) {
        System.out.println("订单未发货，无法收货");
    }
    
    @Override
    public void cancel(Order order) {
        System.out.println("订单已取消");
        order.setState(new CancelledState());
    }
    
    @Override
    public String getStateName() { return "待支付"; }
}

// 具体状态 - 已支付
public class PaidState implements OrderState {
    @Override
    public void pay(Order order) {
        System.out.println("订单已支付");
    }
    
    @Override
    public void ship(Order order) {
        System.out.println("发货成功");
        order.setState(new ShippedState());
    }
    
    @Override
    public void receive(Order order) {
        System.out.println("订单未发货");
    }
    
    @Override
    public void cancel(Order order) {
        System.out.println("退款成功，订单已取消");
        order.setState(new CancelledState());
    }
    
    @Override
    public String getStateName() { return "已支付"; }
}

// 具体状态 - 已发货
public class ShippedState implements OrderState {
    @Override
    public void pay(Order order) { System.out.println("订单已支付"); }
    @Override
    public void ship(Order order) { System.out.println("订单已发货"); }
    
    @Override
    public void receive(Order order) {
        System.out.println("确认收货");
        order.setState(new CompletedState());
    }
    
    @Override
    public void cancel(Order order) {
        System.out.println("订单已发货，无法取消");
    }
    
    @Override
    public String getStateName() { return "已发货"; }
}

// 具体状态 - 已完成
public class CompletedState implements OrderState {
    @Override
    public void pay(Order order) { System.out.println("订单已完成"); }
    @Override
    public void ship(Order order) { System.out.println("订单已完成"); }
    @Override
    public void receive(Order order) { System.out.println("订单已完成"); }
    @Override
    public void cancel(Order order) { System.out.println("订单已完成，无法取消"); }
    @Override
    public String getStateName() { return "已完成"; }
}

// 具体状态 - 已取消
public class CancelledState implements OrderState {
    @Override
    public void pay(Order order) { System.out.println("订单已取消"); }
    @Override
    public void ship(Order order) { System.out.println("订单已取消"); }
    @Override
    public void receive(Order order) { System.out.println("订单已取消"); }
    @Override
    public void cancel(Order order) { System.out.println("订单已取消"); }
    @Override
    public String getStateName() { return "已取消"; }
}

// 上下文
public class Order {
    private OrderState state;
    private String orderId;
    
    public Order(String orderId) {
        this.orderId = orderId;
        this.state = new PendingState();
    }
    
    public void setState(OrderState state) {
        this.state = state;
        System.out.println("订单状态变更: " + state.getStateName());
    }
    
    public void pay() { state.pay(this); }
    public void ship() { state.ship(this); }
    public void receive() { state.receive(this); }
    public void cancel() { state.cancel(this); }
}

// 使用
Order order = new Order("ORD001");
order.pay();     // 待支付 -> 已支付
order.ship();    // 已支付 -> 已发货
order.receive(); // 已发货 -> 已完成
```

---

## 七、其他行为型模式简述

### 7.1 中介者模式
- 用一个中介对象来封装一系列的对象交互
- 典型应用：MVC 框架中的 Controller

### 7.2 备忘录模式
- 在不破坏封装性的前提下，捕获一个对象的内部状态
- 典型应用：Git 的版本回滚

### 7.3 访问者模式
- 表示一个作用于某对象结构中的各元素的操作
- 典型应用：编译器 AST 遍历

### 7.4 迭代器模式
- 提供一种方法顺序访问一个聚合对象中的各个元素
- 典型应用：Java Collection 的 Iterator

---

## 八、模式选择指南

| 需求 | 推荐模式 |
|------|----------|
| 一对多通知 | 观察者模式 |
| 算法可替换 | 策略模式 |
| 算法骨架固定，步骤可变 | 模板方法 |
| 多级审批/过滤 | 责任链模式 |
| 请求参数化/撤销操作 | 命令模式 |
| 状态驱动行为变化 | 状态模式 |

---

> 💡 **提示**：行为型模式关注的是对象间的交互，理解消息传递和职责分配是关键。
