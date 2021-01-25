package org.fisco.bcos.account.client;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import java.util.Properties;
import org.fisco.bcos.account.contract.Account;
import org.fisco.bcos.sdk.BcosSDK;
import org.fisco.bcos.sdk.abi.datatypes.generated.tuples.generated.Tuple2;
import org.fisco.bcos.sdk.client.Client;
import org.fisco.bcos.sdk.crypto.keypair.CryptoKeyPair;
import org.fisco.bcos.sdk.model.TransactionReceipt;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;

public class AccountClient {

  static Logger logger = LoggerFactory.getLogger(AccountClient.class);

  private BcosSDK bcosSDK;
  private Client client;
  private CryptoKeyPair cryptoKeyPair;

  //initialize BcosSDK
  public void initialize() throws Exception {
    @SuppressWarnings("resource")
    ApplicationContext context =
        new ClassPathXmlApplicationContext("classpath:applicationContext.xml");
    bcosSDK = context.getBean(BcosSDK.class);
    client = bcosSDK.getClient(1);
    cryptoKeyPair = client.getCryptoSuite().createKeyPair();
    client.getCryptoSuite().setCryptoKeyPair(cryptoKeyPair);
    logger.debug("create client for group1, account address is " + cryptoKeyPair.getAddress());
  }

  public void deployAccountAndRecordAddr() {

    try {
      Account account = Account.deploy(client, cryptoKeyPair);
      System.out.println(
          " deploy Account success, contract address is " + account.getContractAddress());

      recordAccountAddr(account.getContractAddress());
    } catch (Exception e) {
      // TODO Auto-generated catch block
      // e.printStackTrace();
      System.out.println(" deploy Account contract failed, error message is  " + e.getMessage());
    }
  }

  public void recordAccountAddr(String address) throws FileNotFoundException, IOException {
    Properties prop = new Properties();
    prop.setProperty("address", address);
    final Resource contractResource = new ClassPathResource("contract.properties");
    FileOutputStream fileOutputStream = new FileOutputStream(contractResource.getFile());
    prop.store(fileOutputStream, "contract address");
  }

  public String loadAccountAddr() throws Exception {
    // load Account contact address from contract.properties
    Properties prop = new Properties();
    final Resource contractResource = new ClassPathResource("contract.properties");
    prop.load(contractResource.getInputStream());

    String contractAddress = prop.getProperty("address");
    if (contractAddress == null || contractAddress.trim().equals("")) {
      throw new Exception(" load Account contract address failed, please deploy it first. ");
    }
    logger.info(" load Account address from contract.properties, address is {}", contractAddress);
    return contractAddress;
  }

  public void queryAssetValue(String assetAccount) {
    try {
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      Tuple2<BigInteger, BigInteger> result = account.select(assetAccount);
      if (result.getValue1().compareTo(new BigInteger("0")) == 0) {
        System.out.printf(" Account %s, value %s \n", assetAccount, result.getValue2());
      } else {
        System.out.printf(" %s Account is not exist \n", assetAccount);
      }
    } catch (Exception e) {
      // TODO Auto-generated catch block
      // e.printStackTrace();
      logger.error(" queryAssetValue exception, error message is {}", e.getMessage());

      System.out.printf(" queryAssetValue failed, error message is %s\n", e.getMessage());
    }
  }

  public void queryTransaction(String id){
    try{
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      Tuple2<List<BigInteger>, List<byte[]> > result = account.select_transaction(id);
      if (result.getValue1().get(0).compareTo(new BigInteger("0")) == 0){
        String acc1 = new String(result.getValue2().get(0));
        String acc2 = new String(result.getValue2().get(1));
        System.out.printf(" ID %s, Transaction %s %s %s %s\n", id, acc1, acc2, result.getValue1().get(1), result.getValue1().get(2));
      }
      else{
        System.out.printf(" %s ID is not exist\n", id);
      }
    }
    catch(Exception e){
      logger.error("queryTransaction exception, error message is {}", e.getMessage());
      System.out.printf("queryTransaction failed, error message is %s\n", e.getMessage());
    }
  }

  public void registerAccount(String assetAccount, BigInteger amount) {
    try {
      String contractAddress = loadAccountAddr();

      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      TransactionReceipt receipt = account.register(assetAccount, amount);
      List<Account.RegisterEventEventResponse> response = account.getRegisterEventEvents(receipt);
      if (!response.isEmpty()) {
        if (response.get(0).ret.compareTo(new BigInteger("0")) == 0) {
          System.out.printf(
              " register Account success => Account: %s, value: %s \n", assetAccount, amount);
        } else {
          System.out.printf(
              " register Account failed, ret code is %s \n", response.get(0).ret.toString());
        }
      } else {
        System.out.println(" event log not found, maybe transaction not exec. ");
      }
    } catch (Exception e) {
      // TODO Auto-generated catch block
      // e.printStackTrace();

      logger.error(" registerAccount exception, error message is {}", e.getMessage());
      System.out.printf(" registerAccount failed, error message is %s\n", e.getMessage());
    }
  }

  public void transferAccount(String fromAccount, String toAccount, BigInteger amount) {
    try {
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      TransactionReceipt receipt = account.transfer(fromAccount, toAccount, amount);
      List<Account.TransferEventEventResponse> response = account.getTransferEventEvents(receipt);
      if (!response.isEmpty()) {
        if (response.get(0).ret.compareTo(new BigInteger("0")) == 0) {
          System.out.printf(
              " transfer success => from_Account: %s, to_Account: %s, amount: %s \n",
              fromAccount, toAccount, amount);
        } else {
          System.out.printf(
              " transfer Account failed, ret code is %s \n", response.get(0).ret.toString());
        }
      } else {
        System.out.println(" event log not found, maybe transaction not exec. ");
      }
    } catch (Exception e) {
      // TODO Auto-generated catch block
      // e.printStackTrace();

      logger.error(" transferAccount exception, error message is {}", e.getMessage());
      System.out.printf(" transferAccount failed, error message is %s\n", e.getMessage());
    }
  }

  public void addTransaction(String id, String acc1, String acc2, BigInteger money){
    try{
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      TransactionReceipt receipt = account.addTransaction(id, acc1, acc2, money);
      List<Account.AddTransactionEventEventResponse> response = account.getAddTransactionEventEvents(receipt);
      if (!response.isEmpty()){
        if (response.get(0).ret.compareTo(new BigInteger("0")) == 0){
          System.out.printf("addTransaction success: id %s, acc1 %s, acc2 %s, money %d\n", id, acc1, acc2, money);
        }
        else{
          System.out.printf("addTransaction failed, ret code is %s\n", response.get(0).ret.toString());
        }
      }
      else{
        System.out.printf(" event log not found, maybe transaction not exec. ");
      }
    }
    catch (Exception e){
      logger.error(" addTransaction exception, error message is {}", e.getMessage());
      System.out.printf(" addTransaction failed, error message is %s\n", e.getMessage());
    }
  }

  public void updateTransaction(String id, BigInteger money){
    try{
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      TransactionReceipt receipt = account.updateTransaction(id, money);
      List<Account.UpdateTransactionEventEventResponse> response = account.getUpdateTransactionEventEvents(receipt);
      if (!response.isEmpty()){
        if (response.get(0).ret.compareTo(new BigInteger("0")) == 0){
          System.out.printf("updateTransaction success: id %s, money %d\n", id, money);
        }
        else{
          System.out.printf("updateTransaction failed, ret code is %s\n", response.get(0).ret.toString());
        }
      }
      else{
        System.out.printf(" event log not found, maybe transaction not exec. ");
      }
    }
    catch (Exception e){
      logger.error(" updateTransaction exception, error message is {}", e.getMessage());
      System.out.printf(" updateTransaction failed, error message is %s\n", e.getMessage());
    }
  }

  public void splitTransaction(String old_id, String new_id, String acc, BigInteger money){
    try{
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      TransactionReceipt receipt = account.splitTransaction(old_id,new_id,acc,money);
      List<Account.SplitTransactionEventEventResponse> response = account.getSplitTransactionEventEvents(receipt);
      if (!response.isEmpty()){
        if (response.get(0).ret.compareTo(new BigInteger("0")) == 0){
          System.out.printf("splitTransaction success: old id %s, new id %s, acc %s, money %d\n", old_id, new_id, acc, money);
        }
        else{
          System.out.printf("splitTransaction failed, ret code is %s\n", response.get(0).ret.toString());
        }
      }
      else{
        System.out.printf(" event log not found, maybe transaction not exec. ");
      }
    }
    catch (Exception e){
      logger.error(" splitTransaction exception, error message is {}", e.getMessage());
      System.out.printf(" splitTransaction failed, error message is %s\n", e.getMessage());
    }
  }

  public void removeTransaction(String id){
    try{
      String contractAddress = loadAccountAddr();
      Account account = Account.load(contractAddress, client, cryptoKeyPair);
      TransactionReceipt receipt = account.removeTransaction(id);
      List<Account.RemoveTransactionEventEventResponse> response = account.getRemoveTransactionEventEvents(receipt);
      if (!response.isEmpty()){
        if (response.get(0).ret.compareTo(new BigInteger("0")) == 0){
          System.out.printf("removeTransaction success: id %s\n", id);
        }
        else{
          System.out.printf("removeTransaction failed, ret code is %s\n", response.get(0).ret.toString());
        }
      }
      else{
        System.out.printf(" event log not found, maybe transaction not exec. ");
      }
    }
    catch (Exception e){
      logger.error(" removeTransaction exception, error message is {}", e.getMessage());
      System.out.printf(" removeTransaction failed, error message is %s\n", e.getMessage());
    }
  }


  public static void Usage() {
    System.out.println(" Usage:");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient deploy");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient query1 account");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient query2 transaction");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient register account value");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient transfer from_account to_account amount");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient add_transaction id acc1 acc2 money");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient update_transaction id money");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient split_transaction old_id new_id acc money");
    System.out.println(
        "\t java -cp conf/:lib/*:apps/* org.fisco.bcos.Account.client.AccountClient remove_transaction id");
    System.exit(0);
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 1) {
      Usage();
    }

    AccountClient client = new AccountClient();
    client.initialize();

    switch (args[0]) {
      case "deploy":
        client.deployAccountAndRecordAddr();
        break;
      case "select":
        if (args.length < 2) {
          Usage();
        }
        client.queryAssetValue(args[1]);
        break;
      case "select_transaction":
        if (args.length < 2){
          Usage();
        }
        client.queryTransaction(args[1]);
        break;
      case "register":
        if (args.length < 3) {
          Usage();
        }
        client.registerAccount(args[1], new BigInteger(args[2]));
        break;
      case "transfer":
        if (args.length < 4) {
          Usage();
        }
        client.transferAccount(args[1], args[2], new BigInteger(args[3]));
        break;
      case "addTransaction":
        if (args.length < 5){
          Usage();
        }
        client.addTransaction(args[1], args[2], args[3], new BigInteger(args[4]));
        break;
      case "updateTransaction":
        if (args.length < 3){
          Usage();
        }
        client.updateTransaction(args[1], new BigInteger(args[2]));
        break;
      case "splitTransaction":
        if (args.length < 5){
          Usage();
        }
        client.splitTransaction(args[1], args[2], args[3], new BigInteger(args[4]));
        break;
      case "removeTransaction":
        if (args.length < 2){
          Usage();
        }
        client.removeTransaction(args[1]);
        break;
      default:
        {
          Usage();
        }
    }
    System.exit(0);
  }
}
