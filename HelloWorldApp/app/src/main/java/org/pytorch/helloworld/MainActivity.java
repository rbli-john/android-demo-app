package org.pytorch.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.MemoryFormat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
  private static final String TAG = "oua_" + MainActivity.class.getSimpleName();

  private static final boolean CLASSIFICATION_MODEL = true;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    Bitmap bitmap = null;
    Module module = null;
    try {
      // creating bitmap from packaged into app android asset 'image.jpg',
      // app/src/main/assets/image.jpg
      Bitmap orgBitmap = BitmapFactory.decodeStream(getAssets().open("image.jpg"));

      Log.i(TAG, String.format("The shape of original bitmap (wxh): %dx%d", orgBitmap.getWidth(), orgBitmap.getHeight()));

      if (CLASSIFICATION_MODEL) {
        Log.i(TAG, "Classification model");
        bitmap = orgBitmap;
      } else {
        final int inputWidth = 256;
        final int inputHeight = 256;
        bitmap = Bitmap.createScaledBitmap(orgBitmap, inputWidth, inputHeight, true);
      }

      Log.i(TAG, String.format("The shape of resized bitmap (wxh): %dx%d", bitmap.getWidth(), bitmap.getHeight()));

      // Seems that the default numThreads are not 1, because, by experiments, default number
      // works almost as good as numThreads == 4 (which is best by my experiments).
      // So, I guess pytorch andriod use a good default number.
      PyTorchAndroid.setNumThreads(4);

      // loading serialized torchscript module from packaged into app android asset model.pt,
      // app/src/model/assets/model.pt
      module = LiteModuleLoader.load(
              assetFilePath(this, "mobilenet_v2.pt"),
              null,
              Device.CPU);
      Log.i(TAG, "After load model");
    } catch (IOException e) {
      Log.e(TAG, "Error reading assets", e);
      finish();
    } catch (Exception e) {
      Log.e(TAG, "Unexpected exception");
      Log.e(TAG, e.getMessage(), e);
      finish();
    }

    // showing image on UI
    ImageView imageView = findViewById(R.id.image);
    imageView.setImageBitmap(bitmap);

    // preparing input tensor
    final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

    Log.i(TAG, "input_shape=" + Arrays.toString(inputTensor.shape()));

    // running the model
    Tensor outputTensor = null;
    for (int k=0; k<50; k++) {
      outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
    }

    Log.i(TAG, "inference done");

    // getting tensor content as java array of floats
    final float[] output = outputTensor.getDataAsFloatArray();
    Log.i(TAG, "output: " + Arrays.toString(output));

    if (!CLASSIFICATION_MODEL) {
      ((TextView) findViewById(R.id.text)).setText("Done");
      return;
    }

    final float[] scores = output;

    // searching for the index with maximum score
    float maxScore = -Float.MAX_VALUE;
    int maxScoreIdx = -1;
    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxScoreIdx = i;
      }
    }

    String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

    // showing className on UI
    TextView textView = findViewById(R.id.text);
    textView.setText(className);
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
