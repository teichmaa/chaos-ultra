package cz.cuni.mff.cgg.teichmaa.chaosultra.util;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * @throws com.google.gson.JsonSyntaxException
 */
public class JsonHelpers {
    public static JsonObject parse(String json){
        return new JsonParser().parse(json).getAsJsonObject();
    }

    public static <T> List<T> jsonArrayToList(JsonArray jsonArray, Function<? super JsonElement, ? extends T> mapper){
        return StreamSupport.stream(jsonArray.spliterator(), false).map(mapper).collect(Collectors.toList());
    }

    public static List<Integer> jsonArrayToIntList(JsonArray jsonArray, Function<? super JsonElement, ? extends Integer> mapper){
        return StreamSupport.stream(jsonArray.spliterator(), false).map(mapper).collect(Collectors.toList());
    }
}
