package cz.cuni.mff.cgg.teichmaa.chaosultra.gui;

public interface GUIPresenter {
    /**
     * callback that has to be called by the controller each time a model is updated and should be visualized to the user
     *
     * @param model the updated model
     */
    void onModelUpdated(GUIModel model);

    /**
     * Asynchronously show a blocking alert to the user, that will block any user interaction until confirmed. This method returns right after call.
     *
     * @param message message to show
     */
    void showBlockingErrorAlertAsync(String message);
}
